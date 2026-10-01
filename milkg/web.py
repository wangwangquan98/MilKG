"""Local FastAPI service for the Vue MilKG-QA workbench."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
import time
import urllib.request
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from urllib.parse import urlencode, urlparse

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator

from .augmentation import materialize_specifications
from .documents import Chunk, load_chunks
from .extraction import extract_chunk
from .graph import MilitaryGraph
from .llm import ChatModel, OpenAICompatibleModel, RequestCancelled, is_local_endpoint
from .neo4j_store import Neo4jGraphStore, document_rows
from .qa import QUESTION_TYPES, export_sft, generate_qa
from .traversal import atomic_facts, traverse


ROOT = Path(__file__).resolve().parent.parent
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_UPLOAD_FILES = 200
MAX_TOTAL_UPLOAD_BYTES = 200 * 1024 * 1024
FORMATS = {"alpaca", "sharegpt", "chatml"}
RELATION_REJECTION_LABELS = {
    "unknown_endpoint": "实体未通过校验",
    "evidence_not_in_source": "证据不在原文",
    "invalid_type_or_direction": "关系类型或方向不符",
    "low_confidence": "置信度不足",
    "no_development_statement": "缺少明确研发描述",
    "generic_developer": "国别或朝代不是研发机构",
    "no_counter_statement": "缺少明确克制或打击描述",
    "endpoint_not_in_evidence": "单位或具体装备未出现在关系证据中",
    "invalid_record": "记录格式错误",
}
MODEL_HEARTBEAT_SECONDS = 30
job_logger = logging.getLogger("milkg.web.jobs")
if not job_logger.handlers:
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(logging.Formatter("%(message)s"))
    job_logger.addHandler(stream)
job_logger.setLevel(logging.INFO)
job_logger.propagate = False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _display_name(value: str) -> str:
    parts = [re.sub(r"[\x00-\x1f\x7f]", "", part).strip()
             for part in value.replace("\\", "/").split("/")]
    parts = [part for part in parts if part and part not in {".", ".."}]
    return "/".join(parts) or "document.txt"


class RunConfig(BaseModel):
    mode: Literal["run", "build", "generate"] = "run"
    storage: Literal["json", "neo4j"] = "json"
    graph_action: Literal["new", "extend"] = "new"
    graph_id: str | None = None
    graph_name: str = Field(default="MilKG", max_length=120)
    neo4j_uri: str = "bolt://127.0.0.1:7687"
    neo4j_user: str = "neo4j"
    neo4j_database: str = "neo4j"
    api_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    extract_model: str = Field(default="qwen3.5-flash", min_length=1, max_length=100)
    generate_model: str = Field(default="qwen3.5-plus", min_length=1, max_length=100)
    extract_temperature: float = Field(default=0.1, ge=0.0, le=1.5)
    generate_temperature: float = Field(default=0.7, ge=0.0, le=1.5)
    local_thinking: bool = False
    local_max_tokens: int = Field(default=4096, ge=512, le=16384)
    max_chars: int = Field(default=1800, ge=400, le=8000)
    overlap: int = Field(default=180, ge=0, le=1000)
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    max_subgraphs: int = Field(default=24, ge=1)
    max_atomic: int = Field(default=24, ge=0)
    per_subgraph: int = Field(default=1, ge=1)
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
        if self.mode == "generate" and self.storage != "neo4j":
            raise ValueError("WebUI 的独立生成模式需要 Neo4j 图谱")
        if (self.mode == "generate" or self.graph_action == "extend") and self.storage == "neo4j" and not self.graph_id:
            raise ValueError("请选择已有的 Neo4j 图谱")
        if self.graph_action == "extend" and self.storage != "neo4j":
            raise ValueError("WebUI 的扩展图谱模式需要 Neo4j")
        if self.storage == "neo4j":
            parsed = urlparse(self.neo4j_uri)
            if parsed.scheme not in {"bolt", "bolt+s", "bolt+ssc", "neo4j", "neo4j+s", "neo4j+ssc"}:
                raise ValueError("Neo4j 地址须使用 bolt:// 或 neo4j:// 协议")
            if parsed.username or parsed.password:
                raise ValueError("Neo4j 地址中不要包含用户名或密码，请使用单独的输入框")
        return self


@dataclass
class Job:
    id: str
    filename: str
    source_path: Path | None
    output_dir: Path
    config: RunConfig
    source_paths: list[Path] = field(default_factory=list)
    source_names: list[str] = field(default_factory=list)
    status: str = "queued"
    stage: str = "queued"
    progress: int = 0
    current: int = 0
    total: int = 0
    error: str | None = None
    graph_nodes: int = 0
    graph_edges: int = 0
    graph_id: str | None = None
    semantic_subgraphs: int = 0
    atomic_subgraphs: int = 0
    logs: list[dict] = field(default_factory=list)
    items: list[dict] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    log_count: int = 0
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    active_model: ChatModel | None = field(default=None, repr=False)
    future: Future | None = field(default=None, repr=False)

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
            entry = {"time": _now(), "level": level,
                     "stage": self.stage, "message": message}
            self.updated_at = entry["time"]
            self.logs.append(entry)
            self.log_count += 1
            self.logs = self.logs[-500:]
            line = _log_line(self.id, entry)
            with (self.output_dir / "process.log").open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            _write_json(self.output_dir / "job_state.json", self.snapshot())
            job_logger.log(logging.ERROR if level == "error" else
                           logging.WARNING if level == "warning" else logging.INFO, line)

    def snapshot(self) -> dict:
        with self.lock:
            return {"id": self.id, "filename": self.filename,
                    "sources": list(self.source_names), "file_count": len(self.source_paths),
                    "status": self.status,
                    "resumable": self.status in {"failed", "cancelled"} and self.graph_id is None
                    and self.total > 0 and self.current < self.total
                    and (self.output_dir / "chunks.json").exists()
                    and not (self.output_dir / "graph.json").exists(),
                    "stage": self.stage, "progress": self.progress,
                    "current": self.current, "total": self.total,
                    "created_at": self.created_at, "updated_at": self.updated_at,
                    "error": self.error, "logs": list(self.logs),
                    "log_count": self.log_count, "item_count": len(self.items),
                    "graph_nodes": self.graph_nodes, "graph_edges": self.graph_edges,
                    "graph_id": self.graph_id or self.config.graph_id,
                    "mode": self.config.mode, "storage": self.config.storage,
                    "api_url": self.config.api_url,
                    "local_thinking": self.config.local_thinking,
                    "local_max_tokens": self.config.local_max_tokens,
                    "semantic_subgraphs": self.semantic_subgraphs,
                    "atomic_subgraphs": self.atomic_subgraphs,
                    "output_format": self.config.output_format}


class CheckedModel:
    def __init__(self, client: ChatModel, job: Job):
        self.client = client
        self.job = job

    def complete(self, system: str, user: str, temperature: float) -> dict:
        with self.job.lock:
            if self.job.cancel_event.is_set():
                raise RequestCancelled()
            self.job.active_model = self.client
        done = threading.Event()
        started = time.monotonic()

        def heartbeat() -> None:
            while not done.wait(MODEL_HEARTBEAT_SECONDS):
                elapsed = int(time.monotonic() - started)
                self.job.note(f"模型请求仍在进行，已等待 {elapsed} 秒。")

        watcher = threading.Thread(target=heartbeat, name=f"milkg-heartbeat-{self.job.id[:8]}",
                                   daemon=True)
        watcher.start()
        try:
            result = self.client.complete(system, user, temperature)
            if self.job.cancel_event.is_set():
                raise RequestCancelled()
            return result
        except Exception as exc:
            if self.job.cancel_event.is_set():
                raise RequestCancelled() from exc
            raise
        finally:
            done.set()
            watcher.join()
            with self.job.lock:
                if self.job.active_model is self.client:
                    self.job.active_model = None


def _log_line(job_id: str, entry: dict) -> str:
    return (f"{entry['time']} [{entry['level'].upper()}] "
            f"[{job_id[:8]}] [{entry['stage']}] {entry['message']}")


class JobManager:
    def __init__(self, output_root: Path | None = None, max_workers: int = 2):
        self.output_root = output_root or Path(os.getenv("MILKG_WEB_OUTPUT", ROOT / "output" / "webui"))
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="milkg-job")

    def create(self, filename: str, content: bytes | None | list[tuple[str, bytes]], config: RunConfig,
               api_key: str, start: bool = True, neo4j_password: str = "") -> Job:
        job_id = uuid.uuid4().hex
        output_dir = self.output_root / job_id
        output_dir.mkdir(parents=True)
        uploads = ([(filename, content)] if isinstance(content, bytes) else content) or []
        source_paths, source_names, source_rows = [], [], []
        if uploads:
            upload_dir = output_dir / "uploads"
            upload_dir.mkdir()
            for index, (name, data) in enumerate(uploads, 1):
                display_name = _display_name(name)
                safe_name = re.sub(r"[^\w.\-]", "_", display_name.rsplit("/", 1)[-1],
                                   flags=re.UNICODE).strip("._") or "document.txt"
                stored_name = f"{index:04d}_{safe_name}"
                source = upload_dir / stored_name
                source.write_bytes(data)
                source_paths.append(source)
                source_names.append(display_name)
                source_rows.append({"name": display_name, "stored_name": stored_name, "size": len(data)})
            _write_json(output_dir / "sources.json", source_rows)
        source_path = source_paths[0] if source_paths else None
        _write_json(output_dir / "config.json", config.model_dump())
        title = f"{len(source_paths)} 份文档" if len(source_paths) > 1 else filename
        job = Job(job_id, title, source_path, output_dir, config,
                  source_paths=source_paths, source_names=source_names)
        job.note("图谱任务已创建，等待处理。" if config.mode == "generate" else
                 f"已接收 {len(source_paths)} 份文档，等待处理。")
        with self.lock:
            self.jobs[job_id] = job
        if start:
            job.future = self.executor.submit(self.run, job, api_key, neo4j_password)
        return job

    def cancel(self, job: Job) -> None:
        with job.lock:
            if job.status == "cancelling":
                return
            if job.status not in {"queued", "running"}:
                raise HTTPException(status_code=409, detail="该任务已结束，无法中断")
            job.cancel_event.set()
            job.status = "cancelling"
            active_model = job.active_model
            future = job.future
        job.note("已收到中断请求，正在停止当前步骤。", stage="cancelling")
        if future is not None and future.cancel():
            with job.lock:
                job.status = "cancelled"
            job.note("任务已中断。", stage="cancelled")
        elif active_model is not None:
            stop = getattr(active_model, "cancel", None)
            if callable(stop):
                stop()

    def resume(self, job: Job, api_key: str, neo4j_password: str = "",
               local_thinking: bool | None = None,
               local_max_tokens: int | None = None) -> None:
        with job.lock:
            if not job.snapshot()["resumable"]:
                raise HTTPException(status_code=409, detail="该任务没有可继续的抽取检查点")
            if is_local_endpoint(job.config.api_url):
                updates = {}
                if local_thinking is not None:
                    updates["local_thinking"] = local_thinking
                if local_max_tokens is not None:
                    updates["local_max_tokens"] = local_max_tokens
                if updates:
                    try:
                        settings = RunConfig.model_validate({**job.config.model_dump(), **updates})
                    except ValueError as exc:
                        raise HTTPException(status_code=400, detail=f"本地模型参数无效：{exc}") from exc
                    job.config = settings
                    _write_json(job.output_dir / "config.json", settings.model_dump())
            job.status = "queued"
            job.error = None
            job.cancel_event.clear()
        job.note(f"继续抽取任务已提交，将跳过前 {job.current} 个已完成片段。",
                 stage="queued")
        job.future = self.executor.submit(self.run, job, api_key, neo4j_password, True)

    def get(self, job_id: str) -> Job:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None and re.fullmatch(r"[0-9a-f]{32}", job_id):
                job = self._restore(job_id)
                if job is not None:
                    self.jobs[job_id] = job
        if job is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        return job

    def list_jobs(self, limit: int = 30) -> list[dict]:
        with self.lock:
            active = dict(self.jobs)
        summaries = []
        for output_dir in self.output_root.iterdir():
            if not output_dir.is_dir() or not re.fullmatch(r"[0-9a-f]{32}", output_dir.name):
                continue
            job = active.get(output_dir.name)
            if job is not None:
                state = job.snapshot()
            else:
                state_path = output_dir / "job_state.json"
                try:
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if state.get("status") in {"queued", "running", "cancelling"}:
                    state["status"] = "failed"
                    state["stage"] = "failed"
            summaries.append({key: state.get(key) for key in
                              ("id", "filename", "status", "stage", "progress", "created_at", "updated_at")})
        summaries.sort(key=lambda row: row.get("updated_at") or row.get("created_at") or "", reverse=True)
        return summaries[:limit]

    def _restore(self, job_id: str) -> Job | None:
        output_dir = self.output_root / job_id
        config_path = output_dir / "config.json"
        if not config_path.is_file():
            return None
        try:
            config = RunConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
            manifest = output_dir / "sources.json"
            if manifest.exists():
                source_rows = json.loads(manifest.read_text(encoding="utf-8"))
                source_paths = [output_dir / "uploads" / row["stored_name"] for row in source_rows]
                source_names = [row["name"] for row in source_rows]
                if any(not path.is_file() or path.parent != output_dir / "uploads"
                       for path in source_paths):
                    return None
            else:
                source_paths = [path for path in output_dir.iterdir()
                                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS]
                source_names = [path.name for path in source_paths]
            source = source_paths[0] if source_paths else None
            if source is None and config.mode != "generate":
                return None
            state_path = output_dir / "job_state.json"
            state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
            qa_path = output_dir / "qa.json"
            items = json.loads(qa_path.read_text(encoding="utf-8")) if qa_path.exists() else []
            job = Job(job_id, state.get("filename", source.name if source else "已有图谱"),
                      source, output_dir, config,
                      source_paths=source_paths, source_names=source_names)
            job.graph_id = (state.get("graph_id") if (output_dir / "graph_ref.json").exists()
                            or (output_dir / "graph.json").exists() else None)
            job.items = items
            job.created_at = state.get("created_at", job.created_at)
            job.updated_at = state.get("updated_at", job.created_at)
            job.logs = state.get("logs", [])[-500:]
            job.log_count = state.get("log_count", len(job.logs))
            log_path = output_dir / "process.log"
            if not log_path.exists() and job.logs:
                log_path.write_text("".join(_log_line(job_id, entry) + "\n" for entry in job.logs),
                                    encoding="utf-8")
            completed = state.get("status") == "completed" and (
                (output_dir / "graph.json").exists() if config.mode == "build" else
                all((output_dir / f"sft_{fmt}.json").exists() for fmt in FORMATS))
            cancelled = not completed and state.get("status") == "cancelled"
            job.status = "completed" if completed else "cancelled" if cancelled else "failed"
            job.stage = job.status
            job.progress = 100 if completed else state.get("progress", 0)
            job.current = len(items) if completed else state.get("current", 0)
            job.total = len(items) if completed else state.get("total", 0)
            job.error = None if completed or cancelled else (state.get("error") if state.get("status") == "failed"
                                                              else "任务因服务重启而中断，请重新提交。")
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
            if not completed and state.get("status") in {"queued", "running", "cancelling"}:
                job.note(job.error, level="error", stage="failed", progress=job.progress)
            return job
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _model(self, config: RunConfig, name: str, api_key: str, job: Job) -> CheckedModel:
        local = is_local_endpoint(config.api_url)
        return CheckedModel(OpenAICompatibleModel(config.api_url, name, api_key,
                                                   timeout=300 if local else 90,
                                                   enable_thinking=None if local else False,
                                                   reasoning_effort=("none" if local and not config.local_thinking
                                                                     else None),
                                                   max_tokens=config.local_max_tokens if local else None,
                                                   cancel_event=job.cancel_event), job)

    def _neo4j(self, config: RunConfig, password: str) -> Neo4jGraphStore:
        return Neo4jGraphStore(config.neo4j_uri, config.neo4j_user,
                               password or os.getenv("MILKG_NEO4J_PASSWORD", ""),
                               config.neo4j_database)

    def run(self, job: Job, api_key: str, neo4j_password: str = "",
            resume_extractions: bool = False) -> None:
        config = job.config
        store = None
        def check_cancelled() -> None:
            if job.cancel_event.is_set():
                raise RequestCancelled()

        try:
            check_cancelled()
            with job.lock:
                job.status = "running"
            job.note("任务开始，正在准备数据源与模型。",
                     stage="graph" if config.mode == "generate" else "reading", progress=1)
            if config.storage == "neo4j":
                store = self._neo4j(config, neo4j_password)
            check_cancelled()
            if config.mode == "generate":
                kg = store.load_graph(config.graph_id)
                check_cancelled()
                job.graph_id = config.graph_id
                kg.save(job.output_dir / "graph.json")
                _write_json(job.output_dir / "graph_ref.json",
                            {"storage": "neo4j", "graph_id": job.graph_id,
                             "database": store.database})
                job.note(f"已载入图谱 {config.graph_id}。", stage="graph", progress=60)
            else:
                job.note("读取文档并进行分块。", stage="reading", progress=2, current=0, total=0)
                paths = job.source_paths or ([job.source_path] if job.source_path else [])
                labels = {str(path): name for path, name in zip(paths, job.source_names)}
                chunks = [Chunk(chunk.id, labels.get(chunk.source, chunk.source), chunk.text)
                          for chunk in load_chunks(paths, config.max_chars, config.overlap)]
                if not chunks:
                    raise ValueError("文档没有可提取的文本")
                if resume_extractions and (job.output_dir / "chunks.json").exists():
                    saved_chunks = json.loads((job.output_dir / "chunks.json").read_text(encoding="utf-8"))
                    if [row.get("id") for row in saved_chunks] != [chunk.id for chunk in chunks]:
                        raise ValueError("文档分块与原任务不一致，无法继续抽取")
                _write_json(job.output_dir / "chunks.json", [chunk.__dict__ for chunk in chunks])
                job.note(f"完成 {len(paths)} 份文档的分块，共 {len(chunks)} 个片段。",
                         stage="extracting", progress=8,
                         current=0, total=len(chunks))
                extractor = self._model(config, config.extract_model, api_key, job)
                if resume_extractions:
                    checkpoint = job.output_dir / "extractions.json"
                    extractions = json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.exists() else []
                    if (not isinstance(extractions, list) or len(extractions) > len(chunks)
                            or any(not isinstance(row, dict) or row.get("chunk_id") != chunks[i].id
                                   for i, row in enumerate(extractions))):
                        raise ValueError("抽取检查点与当前文档分块不匹配，无法继续")
                    job.note(f"已载入 {len(extractions)} 个片段的抽取结果，继续处理剩余片段。",
                             stage="extracting", current=len(extractions), total=len(chunks),
                             progress=8 + round(42 * len(extractions) / len(chunks)))
                else:
                    extractions = []
                for index, chunk in enumerate(chunks, 1):
                    check_cancelled()
                    if index <= len(extractions):
                        continue
                    job.note(f"片段 {index}/{len(chunks)}（{chunk.source}）：正在抽取实体。", stage="extracting")
                    result = extract_chunk(chunk, extractor, config.min_confidence,
                                           config.extract_temperature,
                                           on_phase=lambda message, i=index: job.note(
                                               f"片段 {i}/{len(chunks)}：{message}。", stage="extracting"))
                    check_cancelled()
                    extractions.append(result)
                    _write_json(job.output_dir / "extractions.json", extractions)
                    percent = 8 + round(42 * index / len(chunks))
                    details = result.get("diagnostics", {})
                    job.note(f"片段 {index}/{len(chunks)}：{len(result['entities'])} 个实体，"
                             f"{len(result['relations'])} 条关系（模型候选 "
                             f"{details.get('relation_candidates', len(result['relations']))} 条）。",
                             stage="extracting", progress=percent,
                             current=index, total=len(chunks))
                    reasons = details.get("relation_rejections", {})
                    if reasons:
                        summary = "、".join(
                            f"{RELATION_REJECTION_LABELS.get(reason, reason)} {count} 条"
                            for reason, count in reasons.items()
                        )
                        job.note(f"片段 {index}/{len(chunks)} 关系过滤：{summary}。",
                                 stage="extracting", level="warning")
                    if details.get("focused_pass_error"):
                        job.note(f"片段 {index}/{len(chunks)} 的关系复查请求失败："
                                 f"{details['focused_pass_error']}；已保留首轮结果。",
                                 stage="extracting", level="warning")
                check_cancelled()
                if store:
                    if config.graph_action == "extend":
                        job.graph_id = config.graph_id
                        kg = store.load_graph(job.graph_id)
                        previous = MilitaryGraph.from_dict(kg.to_dict())
                    else:
                        job.graph_id = store.create_graph(config.graph_name)
                        kg, previous = MilitaryGraph(), None
                else:
                    kg, previous = MilitaryGraph(), None
                for extraction in extractions:
                    kg.add_extraction(extraction)
                kg.save(job.output_dir / "graph_raw.json")
                job.note(f"建图完成：{kg.graph.number_of_nodes()} 个节点，"
                         f"{kg.graph.number_of_edges()} 条关系。", stage="graph", progress=55,
                         current=0, total=0)
                if config.materialize_specs:
                    added = materialize_specifications(kg, chunks)
                    job.note(f"补充 {added} 条有原文证据的技术规格关系。", stage="graph", progress=60)
                check_cancelled()
                if store:
                    changes = store.save_graph(job.graph_id, kg, previous, document_rows(chunks))
                    _write_json(job.output_dir / "graph_ref.json",
                                {"storage": "neo4j", "graph_id": job.graph_id,
                                 "database": store.database})
                    job.note(f"Neo4j 图谱 {job.graph_id} 已保存：新增或更新 "
                             f"{changes['changed_nodes']} 个节点、{changes['changed_edges']} 条关系。",
                             stage="graph", progress=62)
                kg.save(job.output_dir / "graph.json")
            check_cancelled()
            with job.lock:
                job.graph_nodes = kg.graph.number_of_nodes()
                job.graph_edges = kg.graph.number_of_edges()
            if config.mode == "build":
                with job.lock:
                    check_cancelled()
                    job.status = "completed"
                job.note(f"构图完成：{job.graph_nodes} 个节点、{job.graph_edges} 条关系。",
                         stage="completed", progress=100)
                return
            def traversal_progress(message: str) -> None:
                check_cancelled()
                job.note(message, stage="traversing", progress=63, current=0, total=0)

            traversal_progress("开始遍历图谱，正在建立关系索引。")
            subgraphs = traverse(kg, max_subgraphs=config.max_subgraphs,
                                 max_paths=max(2000, config.max_subgraphs),
                                 on_progress=traversal_progress)
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
                check_cancelled()
                generator = self._model(config, config.generate_model, api_key, job)
                last_accepted = 0

                def on_progress(items: list[dict], stats: dict) -> None:
                    nonlocal last_accepted
                    check_cancelled()
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
                check_cancelled()
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
                check_cancelled()
                export_sft(items, job.output_dir / f"sft_{fmt}.json", fmt)
            with job.lock:
                check_cancelled()
                job.status = "completed"
            job.note(f"处理完成：{len(items)} 条问答通过校验，可预览和下载。",
                     stage="completed", progress=100, current=len(items), total=len(items))
        except RequestCancelled:
            with job.lock:
                job.status = "cancelled"
                job.error = None
            job.note("任务已中断；已完成的片段和结果保留在任务目录。", stage="cancelled")
        except Exception as exc:
            if job.cancel_event.is_set():
                with job.lock:
                    job.status = "cancelled"
                    job.error = None
                job.note("任务已中断；已完成的片段和结果保留在任务目录。", stage="cancelled")
                return
            message = str(exc).replace(api_key, "[REDACTED]") if api_key else str(exc)
            if neo4j_password:
                message = message.replace(neo4j_password, "[REDACTED]")
            with job.lock:
                job.status = "failed"
                job.error = message[:1000]
            job.note(f"运行失败：{message[:500]}", level="error", stage="failed")
        finally:
            if store:
                store.close()


manager = JobManager()
app = FastAPI(title="MilKG-QA Workbench", version="0.2.0")


class GraphListRequest(BaseModel):
    uri: str = "bolt://127.0.0.1:7687"
    user: str = "neo4j"
    password: str = ""
    database: str = "neo4j"

    @field_validator("uri")
    @classmethod
    def valid_uri(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"bolt", "bolt+s", "bolt+ssc", "neo4j", "neo4j+s", "neo4j+ssc"}:
            raise ValueError("Neo4j 地址须使用 bolt:// 或 neo4j:// 协议")
        if parsed.username or parsed.password:
            raise ValueError("Neo4j 地址中不要包含登录凭据")
        return value


class GraphMigrateRequest(GraphListRequest):
    graph_id: str = Field(min_length=1)


class ModelListRequest(BaseModel):
    api_url: str
    api_key: str = ""

    @field_validator("api_url")
    @classmethod
    def valid_api_url(cls, value: str) -> str:
        return RunConfig.valid_api_url(value)


def _fetch_models(api_url: str, api_key: str) -> list[str]:
    aliyun = (urlparse(api_url).hostname or "").lower().endswith(".aliyuncs.com")
    native = aliyun and api_url.endswith("/compatible-mode/v1")
    endpoint = (api_url[:-len("/compatible-mode/v1")] + "/api/v1/models"
                if native else api_url + "/models")
    names: set[str] = set()
    for page in range(1, 11):
        url = endpoint + ("?" + urlencode({"capabilities": "TG", "page_no": page,
                                           "page_size": 100}) if native else "")
        headers = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.load(response)
        if native:
            output = payload.get("output", {})
            rows = output.get("models", [])
            names.update(row["model"] for row in rows
                         if isinstance(row, dict) and isinstance(row.get("model"), str))
            page_size = output.get("page_size") or len(rows)
            if not rows or page * page_size >= output.get("total", 0):
                break
        else:
            rows = payload.get("data", [])
            names.update(row["id"] for row in rows
                         if isinstance(row, dict) and isinstance(row.get("id"), str))
            break
    if not names:
        raise ValueError("服务未返回可用的文本模型；请检查地址、API Key 或手动填写模型名")
    return sorted(names, key=str.casefold)


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "env_api_key_available": bool(os.getenv("ALIYUN_API_KEY")),
            "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
            "max_upload_files": MAX_UPLOAD_FILES,
            "max_upload_bytes": MAX_UPLOAD_BYTES,
            "max_total_upload_bytes": MAX_TOTAL_UPLOAD_BYTES}


@app.post("/api/models")
def list_models(request: ModelListRequest) -> dict:
    local = is_local_endpoint(request.api_url)
    secret = request.api_key.strip() or ("" if local else os.getenv("ALIYUN_API_KEY", "").strip())
    if not secret and not local:
        raise HTTPException(status_code=400, detail="请填写 API Key，或配置 ALIYUN_API_KEY")
    try:
        return {"models": _fetch_models(request.api_url, secret)}
    except (OSError, ValueError, TypeError, KeyError) as exc:
        message = str(exc).replace(secret, "[REDACTED]") if secret else str(exc)
        raise HTTPException(status_code=400, detail=f"读取模型列表失败：{message[:300]}") from exc


@app.post("/api/graphs/list")
def list_graphs(request: GraphListRequest) -> dict:
    password = request.password or os.getenv("MILKG_NEO4J_PASSWORD", "")
    try:
        store = Neo4jGraphStore(request.uri, request.user, password, request.database)
        try:
            return {"graphs": store.list_graphs()}
        finally:
            store.close()
    except Exception as exc:
        message = str(exc).replace(password, "[REDACTED]") if password else str(exc)
        raise HTTPException(status_code=400, detail=f"Neo4j 连接失败：{message[:300]}") from exc


@app.post("/api/graphs/migrate")
def migrate_graph(request: GraphMigrateRequest) -> dict:
    password = request.password or os.getenv("MILKG_NEO4J_PASSWORD", "")
    try:
        store = Neo4jGraphStore(request.uri, request.user, password, request.database)
        try:
            return store.migrate_graph(request.graph_id)
        finally:
            store.close()
    except Exception as exc:
        message = str(exc).replace(password, "[REDACTED]") if password else str(exc)
        raise HTTPException(status_code=400, detail=f"图谱分类更新失败：{message[:300]}") from exc


@app.post("/api/jobs", status_code=202)
async def create_job(file: UploadFile | None = File(None), files: list[UploadFile] | None = File(None),
                     config: str = Form(...),
                     api_key: str = Form(default=""),
                     neo4j_password: str = Form(default="")) -> dict:
    try:
        settings = RunConfig.model_validate_json(config)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"参数无效：{exc}") from exc
    local = is_local_endpoint(settings.api_url)
    secret = api_key.strip() or ("" if local else os.getenv("ALIYUN_API_KEY", "").strip())
    if not secret and not local:
        raise HTTPException(status_code=400, detail="请填写 API Key，或配置 ALIYUN_API_KEY")
    if settings.mode == "generate":
        filename, content = f"Neo4j 图谱 {settings.graph_id}", None
    else:
        uploads = list(files or []) + ([file] if file else [])
        if not uploads:
            raise HTTPException(status_code=400, detail="请上传至少一份文档")
        if len(uploads) > MAX_UPLOAD_FILES:
            raise HTTPException(status_code=413, detail=f"每批最多 {MAX_UPLOAD_FILES} 份文档")
        content = []
        total_bytes = 0
        for upload in uploads:
            name = _display_name(upload.filename or "")
            if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise HTTPException(status_code=400, detail=f"{name}：仅支持 TXT、MD、PDF、DOCX")
            data = await upload.read(MAX_UPLOAD_BYTES + 1)
            if not data:
                raise HTTPException(status_code=400, detail=f"{name}：上传文件为空")
            if len(data) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail=f"{name}：文件不能超过 20 MB")
            total_bytes += len(data)
            if total_bytes > MAX_TOTAL_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="本批文件总大小不能超过 200 MB")
            content.append((name, data))
        filename = content[0][0] if len(content) == 1 else f"{len(content)} 份文档"
    job = manager.create(filename, content, settings, secret,
                         neo4j_password=neo4j_password)
    return {"id": job.id, "status": job.status}


@app.get("/api/jobs")
def recent_jobs(limit: int = 30) -> dict:
    if not 1 <= limit <= 100:
        raise HTTPException(status_code=400, detail="无效任务数量")
    return {"jobs": manager.list_jobs(limit)}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    return manager.get(job_id).snapshot()


@app.post("/api/jobs/{job_id}/cancel", status_code=202)
def cancel_job(job_id: str) -> dict:
    job = manager.get(job_id)
    manager.cancel(job)
    return {"id": job.id, "status": job.status}


@app.post("/api/jobs/{job_id}/resume", status_code=202)
def resume_job(job_id: str, api_key: str = Form(default=""),
               neo4j_password: str = Form(default=""),
               local_thinking: bool | None = Form(default=None),
               local_max_tokens: int | None = Form(default=None)) -> dict:
    job = manager.get(job_id)
    local = is_local_endpoint(job.config.api_url)
    secret = api_key.strip() or ("" if local else os.getenv("ALIYUN_API_KEY", "").strip())
    if not secret and not local:
        raise HTTPException(status_code=400, detail="请填写 API Key，或配置 ALIYUN_API_KEY")
    manager.resume(job, secret, neo4j_password, local_thinking, local_max_tokens)
    return {"id": job.id, "status": job.status}


@app.get("/api/jobs/{job_id}/logs")
def download_logs(job_id: str) -> FileResponse:
    job = manager.get(job_id)
    path = job.output_dir / "process.log"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="当前任务还没有日志文件")
    return FileResponse(path, media_type="text/plain; charset=utf-8",
                        filename=f"milkg_{job.id[:8]}.log")


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
        if job.config.mode == "build":
            raise HTTPException(status_code=409, detail="当前任务只构建图谱，没有 SFT 数据集")
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
