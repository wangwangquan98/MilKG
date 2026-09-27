"""Local FastAPI service for the Vue MilKG-QA workbench."""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator

from .augmentation import materialize_specifications
from .documents import load_chunks
from .extraction import extract_chunk
from .graph import MilitaryGraph
from .llm import ChatModel, OpenAICompatibleModel
from .qa import QUESTION_TYPES, export_sft, generate_qa
from .traversal import atomic_facts, traverse


ROOT = Path(__file__).resolve().parent.parent
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
FORMATS = {"alpaca", "sharegpt", "chatml"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class RunConfig(BaseModel):
    api_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    extract_model: str = Field(default="qwen3.5-flash", min_length=1, max_length=100)
    generate_model: str = Field(default="qwen3.5-plus", min_length=1, max_length=100)
    extract_temperature: float = Field(default=0.1, ge=0.0, le=1.5)
    generate_temperature: float = Field(default=0.7, ge=0.0, le=1.5)
    max_chars: int = Field(default=1800, ge=400, le=8000)
    overlap: int = Field(default=180, ge=0, le=1000)
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    max_subgraphs: int = Field(default=24, ge=1, le=200)
    max_atomic: int = Field(default=24, ge=0, le=100)
    per_subgraph: int = Field(default=1, ge=1, le=5)
    include_atomic: bool = True
    materialize_specs: bool = True
    question_types: list[str] = Field(default_factory=lambda: list(QUESTION_TYPES))
    output_format: str = "alpaca"
    seed: int = 42

    @field_validator("api_url")
    @classmethod
    def valid_api_url(cls, value: str) -> str:
        parsed = urlparse(value.strip())
        hostname = (parsed.hostname or "").lower()
        official = parsed.scheme == "https" and (hostname == "aliyuncs.com" or hostname.endswith(".aliyuncs.com"))
        local = parsed.scheme == "http" and hostname in {"localhost", "127.0.0.1"}
        if not (official or local):
            raise ValueError("API 地址须为阿里云 HTTPS 端点或本机测试端点")
        return value.strip().rstrip("/")

    @field_validator("question_types")
    @classmethod
    def valid_question_types(cls, value: list[str]) -> list[str]:
        if not value or len(set(value)) != len(value) or not set(value) <= set(QUESTION_TYPES):
            raise ValueError("请至少选择一种有效且不重复的题型")
        return value

    @field_validator("output_format")
    @classmethod
    def valid_output_format(cls, value: str) -> str:
        if value not in FORMATS:
            raise ValueError("不支持的导出格式")
        return value

    @model_validator(mode="after")
    def valid_chunk_window(self) -> "RunConfig":
        if self.overlap >= self.max_chars:
            raise ValueError("分块重叠量必须小于块长")
        return self


@dataclass
class Job:
    id: str
    filename: str
    source_path: Path
    output_dir: Path
    config: RunConfig
    status: str = "queued"
    stage: str = "queued"
    progress: int = 0
    current: int = 0
    total: int = 0
    error: str | None = None
    graph_nodes: int = 0
    graph_edges: int = 0
    semantic_subgraphs: int = 0
    atomic_subgraphs: int = 0
    logs: list[dict] = field(default_factory=list)
    items: list[dict] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def note(self, message: str, level: str = "info", stage: str | None = None,
             progress: int | None = None, current: int | None = None,
             total: int | None = None) -> None:
        with self.lock:
            if stage is not None:
                self.stage = stage
            if progress is not None:
                self.progress = max(0, min(100, progress))
            if current is not None:
                self.current = current
            if total is not None:
                self.total = total
            self.logs.append({"time": _now(), "level": level,
                              "stage": self.stage, "message": message})
            self.logs = self.logs[-500:]
            _write_json(self.output_dir / "job_state.json", self.snapshot())

    def snapshot(self) -> dict:
        with self.lock:
            return {"id": self.id, "filename": self.filename, "status": self.status,
                    "stage": self.stage, "progress": self.progress,
                    "current": self.current, "total": self.total,
                    "created_at": self.created_at, "error": self.error,
                    "logs": list(self.logs), "item_count": len(self.items),
                    "graph_nodes": self.graph_nodes, "graph_edges": self.graph_edges,
                    "semantic_subgraphs": self.semantic_subgraphs,
                    "atomic_subgraphs": self.atomic_subgraphs,
                    "output_format": self.config.output_format}


class CheckedModel:
    def __init__(self, client: ChatModel, job: Job):
        self.client = client
        self.job = job

    def complete(self, system: str, user: str, temperature: float) -> dict:
        return self.client.complete(system, user, temperature)


class JobManager:
    def __init__(self, output_root: Path | None = None, max_workers: int = 2):
        self.output_root = output_root or Path(os.getenv("MILKG_WEB_OUTPUT", ROOT / "output" / "webui"))
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="milkg-job")

    def create(self, filename: str, content: bytes, config: RunConfig,
               api_key: str, start: bool = True) -> Job:
        job_id = uuid.uuid4().hex
        output_dir = self.output_root / job_id
        output_dir.mkdir(parents=True)
        safe_name = re.sub(r"[^\w.\-]", "_", filename, flags=re.UNICODE).strip("._") or "document.txt"
        source_path = output_dir / safe_name
        source_path.write_bytes(content)
        _write_json(output_dir / "config.json", config.model_dump())
        job = Job(job_id, filename, source_path, output_dir, config)
        job.note("文件已接收，等待处理。")
        with self.lock:
            self.jobs[job_id] = job
        if start:
            self.executor.submit(self.run, job, api_key)
        return job

    def get(self, job_id: str) -> Job:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None and re.fullmatch(r"[0-9a-f]{32}", job_id):
                job = self._restore(job_id)
                if job is not None:
                    self.jobs[job_id] = job
        if job is None:
            raise HTTPException(status_code=404, detail="任务不存在或服务已重启")
        return job

    def _restore(self, job_id: str) -> Job | None:
        output_dir = self.output_root / job_id
        config_path = output_dir / "config.json"
        if not config_path.is_file():
            return None
        source = next((path for path in output_dir.iterdir()
                       if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS), None)
        if source is None:
            return None
        try:
            config = RunConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
            state_path = output_dir / "job_state.json"
            state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
            qa_path = output_dir / "qa.json"
            items = json.loads(qa_path.read_text(encoding="utf-8")) if qa_path.exists() else []
            job = Job(job_id, state.get("filename", source.name), source, output_dir, config)
            job.items = items
            job.created_at = state.get("created_at", job.created_at)
            job.logs = state.get("logs", [])[-500:]
            completed = all((output_dir / f"sft_{fmt}.json").exists() for fmt in FORMATS)
            job.status = "completed" if completed else "failed"
            job.stage = "completed" if completed else "failed"
            job.progress = 100 if completed else state.get("progress", 0)
            job.current = len(items) if completed else state.get("current", 0)
            job.total = len(items) if completed else state.get("total", 0)
            job.error = None if completed else "任务因服务重启而中断，请重新提交。"
            graph_path = output_dir / "graph.json"
            if graph_path.exists():
                graph = MilitaryGraph.load(graph_path)
                job.graph_nodes = graph.graph.number_of_nodes()
                job.graph_edges = graph.graph.number_of_edges()
            subgraphs_path = output_dir / "subgraphs.json"
            if subgraphs_path.exists():
                subgraphs = json.loads(subgraphs_path.read_text(encoding="utf-8"))
                job.semantic_subgraphs = sum(s["strategy"] != "atomic_fact" for s in subgraphs)
                job.atomic_subgraphs = len(subgraphs) - job.semantic_subgraphs
            if not job.logs:
                job.note("已从本地任务文件恢复运行结果。", stage=job.stage,
                         progress=job.progress)
            return job
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _model(self, config: RunConfig, name: str, api_key: str, job: Job) -> CheckedModel:
        return CheckedModel(OpenAICompatibleModel(config.api_url, name, api_key,
                                                   enable_thinking=False), job)

    def run(self, job: Job, api_key: str) -> None:
        config = job.config
        try:
            with job.lock:
                job.status = "running"
            job.note("读取文档并进行分块。", stage="reading", progress=2, current=0, total=0)
            chunks = load_chunks([job.source_path], config.max_chars, config.overlap)
            if not chunks:
                raise ValueError("文档没有可提取的文本")
            _write_json(job.output_dir / "chunks.json", [chunk.__dict__ for chunk in chunks])
            job.note(f"完成分块，共 {len(chunks)} 个片段。", stage="extracting", progress=8,
                     current=0, total=len(chunks))
            extractor = self._model(config, config.extract_model, api_key, job)
            extractions = []
            for index, chunk in enumerate(chunks, 1):
                result = extract_chunk(chunk, extractor, config.min_confidence,
                                       config.extract_temperature)
                extractions.append(result)
                _write_json(job.output_dir / "extractions.json", extractions)
                percent = 8 + round(42 * index / len(chunks))
                job.note(f"片段 {index}/{len(chunks)}：{len(result['entities'])} 个实体，"
                         f"{len(result['relations'])} 条关系。", stage="extracting",
                         progress=percent, current=index, total=len(chunks))
            kg = MilitaryGraph()
            for extraction in extractions:
                kg.add_extraction(extraction)
            kg.save(job.output_dir / "graph_raw.json")
            job.note(f"建图完成：{kg.graph.number_of_nodes()} 个节点，"
                     f"{kg.graph.number_of_edges()} 条关系。", stage="graph", progress=55,
                     current=0, total=0)
            if config.materialize_specs:
                added = materialize_specifications(kg, chunks)
                job.note(f"补充 {added} 条有原文证据的技术规格关系。", stage="graph", progress=60)
            kg.save(job.output_dir / "graph.json")
            with job.lock:
                job.graph_nodes = kg.graph.number_of_nodes()
                job.graph_edges = kg.graph.number_of_edges()
            subgraphs = traverse(kg, max_subgraphs=config.max_subgraphs)
            semantic_count = len(subgraphs)
            if config.include_atomic:
                subgraphs.extend(atomic_facts(kg)[:config.max_atomic])
            with job.lock:
                job.semantic_subgraphs = semantic_count
                job.atomic_subgraphs = len(subgraphs) - semantic_count
            _write_json(job.output_dir / "subgraphs.json", [subgraph.to_dict() for subgraph in subgraphs])
            job.note(f"遍历完成：{semantic_count} 个语义子图，"
                     f"{len(subgraphs) - semantic_count} 个单跳事实。",
                     stage="traversing", progress=65, current=0, total=len(subgraphs))
            if config.max_subgraphs > 1 and semantic_count <= 1:
                advice = []
                if not config.materialize_specs:
                    advice.append("可尝试开启“补充技术规格关系”")
                if not config.include_atomic:
                    advice.append("可尝试开启“加入单跳事实”")
                hint = "；".join(advice)
                job.note(f"图谱有 {job.graph_nodes} 个节点、{job.graph_edges} 条关系边；"
                         f"仅找到 {semantic_count} 个语义子图。“最多语义子图”是上限，"
                         f"不会补齐缺少的关系。{hint}", level="warning", stage="traversing")
            if subgraphs:
                generator = self._model(config, config.generate_model, api_key, job)
                last_accepted = 0

                def on_progress(items: list[dict], stats: dict) -> None:
                    nonlocal last_accepted
                    with job.lock:
                        job.items = list(items)
                    _write_json(job.output_dir / "qa.json", items)
                    _write_json(job.output_dir / "stats.json", stats)
                    count = stats["attempted"]
                    percent = 65 + round(30 * count / (len(subgraphs) * config.per_subgraph))
                    if len(items) > last_accepted:
                        job.note(f"保留第 {len(items)} 题：{items[-1]['question'][:72]}",
                                 stage="generating", progress=percent,
                                 current=count, total=len(subgraphs) * config.per_subgraph)
                    else:
                        job.note(f"第 {count} 次生成未通过校验；已保留 {len(items)} 题。",
                                 level="warning", stage="generating", progress=percent,
                                 current=count, total=len(subgraphs) * config.per_subgraph)
                    last_accepted = len(items)

                items, stats = generate_qa(
                    kg, subgraphs, generator, tuple(config.question_types), config.seed,
                    config.per_subgraph, 0.85, on_progress=on_progress,
                    temperature=config.generate_temperature,
                )
            else:
                items, stats = [], {"attempted": 0, "accepted": 0, "invalid": 0,
                                    "duplicates": 0, "reason": "没有可用子图"}
                job.note("图谱没有可遍历关系，导出空数据集。", level="warning",
                         stage="generating", progress=95)
            with job.lock:
                job.items = list(items)
            _write_json(job.output_dir / "qa.json", items)
            _write_json(job.output_dir / "stats.json", stats)
            job.note("正在导出 SFT 数据集。", stage="exporting", progress=97)
            for fmt in sorted(FORMATS):
                export_sft(items, job.output_dir / f"sft_{fmt}.json", fmt)
            with job.lock:
                job.status = "completed"
            job.note(f"处理完成：{len(items)} 条问答通过校验，可预览和下载。",
                     stage="completed", progress=100, current=len(items), total=len(items))
        except Exception as exc:
            message = str(exc).replace(api_key, "[REDACTED]") if api_key else str(exc)
            with job.lock:
                job.status = "failed"
                job.error = message[:1000]
            job.note(f"运行失败：{message[:500]}", level="error", stage="failed")


manager = JobManager()
app = FastAPI(title="MilKG-QA Workbench", version="0.2.0")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "env_api_key_available": bool(os.getenv("ALIYUN_API_KEY")),
            "supported_extensions": sorted(SUPPORTED_EXTENSIONS)}


@app.post("/api/jobs", status_code=202)
async def create_job(file: UploadFile = File(...), config: str = Form(...),
                     api_key: str = Form(default="")) -> dict:
    filename = file.filename or ""
    if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="仅支持 TXT、MD、PDF、DOCX")
    try:
        settings = RunConfig.model_validate_json(config)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"参数无效：{exc}") from exc
    secret = api_key.strip() or os.getenv("ALIYUN_API_KEY", "").strip()
    if not secret:
        raise HTTPException(status_code=400, detail="请填写 API Key，或配置 ALIYUN_API_KEY")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="上传文件为空")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="文件不能超过 20 MB")
    job = manager.create(filename, content, settings, secret)
    return {"id": job.id, "status": job.status}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    return manager.get(job_id).snapshot()


@app.get("/api/jobs/{job_id}/items")
def job_items(job_id: str, offset: int = 0, limit: int = 20, type: str | None = None) -> dict:
    job = manager.get(job_id)
    if offset < 0 or not 1 <= limit <= 100:
        raise HTTPException(status_code=400, detail="无效分页参数")
    if type is not None and type not in QUESTION_TYPES:
        raise HTTPException(status_code=400, detail="无效题型")
    with job.lock:
        items = [item for item in job.items if item["type"] == type] if type else job.items
        return {"total": len(items), "offset": offset, "items": items[offset:offset + limit]}


@app.get("/api/jobs/{job_id}/download")
def download(job_id: str, format: str = "alpaca") -> FileResponse:
    job = manager.get(job_id)
    if format not in FORMATS:
        raise HTTPException(status_code=400, detail="不支持的导出格式")
    with job.lock:
        if job.status != "completed":
            raise HTTPException(status_code=409, detail="任务尚未完成")
    path = job.output_dir / f"sft_{format}.json"
    return FileResponse(path, media_type="application/json",
                        filename=f"milkg_{job.id[:8]}_{format}.json")


FRONTEND_DIST = ROOT / "frontend" / "dist"
if FRONTEND_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="frontend-assets")

    @app.get("/")
    @app.get("/index.html")
    def frontend() -> FileResponse:
        return FileResponse(FRONTEND_DIST / "index.html")

    @app.get("/favicon.svg")
    def favicon() -> FileResponse:
        return FileResponse(FRONTEND_DIST / "favicon.svg", media_type="image/svg+xml")


def main() -> None:
    import uvicorn

    uvicorn.run("milkg.web:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
