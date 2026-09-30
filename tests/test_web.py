import json
import io
import logging
import threading
import time

from fastapi.testclient import TestClient

from milkg import llm, web
from milkg.llm import RequestCancelled


def test_web_job_processes_upload_and_exports(tmp_path):
    class FakeModel:
        def complete(self, system, user, temperature):
            if "实体关系抽取器" in system:
                assert temperature == 0.2
                return {
                    "entities": [
                        {"name": "甲平台", "type": "Platform/Carrier"},
                        {"name": "甲武器", "type": "Weapon System"},
                    ],
                    "relations": [{"source": "甲平台", "target": "甲武器",
                                   "type": "Equip-Carry", "evidence": "甲平台搭载甲武器"}],
                }
            assert temperature == 0.6
            payload = json.loads(user.split("子图：\n", 1)[1].split("\n可用于选项", 1)[0])
            source = next(node["id"] for node in payload["nodes"] if node["name"] == "甲平台")
            answer = next(node["id"] for node in payload["nodes"] if node["name"] == "甲武器")
            return {"question": "甲平台搭载的武器是____。", "options": [],
                    "answer": "甲武器", "explanation": "甲平台搭载甲武器。",
                    "supporting_facts": [payload["facts"][0]["id"]],
                    "referenced_entities": [source, answer], "answer_entities": [answer]}

    manager = web.JobManager(tmp_path)
    manager._model = lambda *args: FakeModel()
    config = web.RunConfig(extract_temperature=0.2, generate_temperature=0.6,
                           question_types=["fill_blank"], include_atomic=True,
                           max_atomic=1, materialize_specs=False)
    job = manager.create("source.txt", "甲平台搭载甲武器。".encode(), config, "secret", start=False)
    manager.run(job, "secret")
    assert job.status == "completed"
    assert job.progress == 100
    assert job.snapshot()["graph_nodes"] == 2
    assert job.snapshot()["graph_edges"] == 1
    assert job.snapshot()["atomic_subgraphs"] == 1
    assert any("复查关系" in entry["message"] for entry in job.logs)
    assert job.items[0]["answer"] == "甲武器"
    assert "secret" not in (job.output_dir / "config.json").read_text(encoding="utf-8")
    assert (job.output_dir / "sft_alpaca.json").exists()
    assert (job.output_dir / "sft_sharegpt.json").exists()
    assert (job.output_dir / "sft_chatml.json").exists()
    assert (job.output_dir / "job_state.json").exists()
    assert (job.output_dir / "process.log").exists()
    restored = web.JobManager(tmp_path).get(job.id)
    assert restored.status == "completed"
    assert restored.snapshot()["item_count"] == 1
    assert restored.snapshot()["graph_edges"] == 1

    original = web.manager
    web.manager = manager
    try:
        client = TestClient(web.app)
        assert client.get(f"/api/jobs/{job.id}").json()["item_count"] == 1
        assert client.get(f"/api/jobs/{job.id}/items").json()["items"][0]["type"] == "fill_blank"
        assert client.get(f"/api/jobs/{job.id}/items?type=fill_blank").json()["total"] == 1
        assert client.get(f"/api/jobs/{job.id}/items?type=cot").json()["total"] == 0
        assert client.get(f"/api/jobs/{job.id}/download?format=alpaca").status_code == 200
        assert client.post("/api/jobs", data={"config": config.model_dump_json()},
                           files={"file": ("source.exe", b"bad")}).status_code == 400
    finally:
        web.manager = original


def test_running_job_is_listed_and_logs_survive_browser_reopen(tmp_path, request):
    console = io.StringIO()
    handler = logging.StreamHandler(console)
    web.job_logger.addHandler(handler)
    request.addfinalizer(lambda: web.job_logger.removeHandler(handler))
    manager = web.JobManager(tmp_path)
    job = manager.create("source.txt", b"sample", web.RunConfig(mode="build"),
                         "secret", start=False)
    with job.lock:
        job.status = "running"
    job.note("片段 2/9：正在抽取实体。", stage="extracting", progress=17,
             current=2, total=9)
    assert "2/9" in console.getvalue()
    assert "片段 2/9" in (job.output_dir / "process.log").read_text(encoding="utf-8")
    original = web.manager
    web.manager = manager
    try:
        client = TestClient(web.app)
        listed = client.get("/api/jobs").json()["jobs"]
        assert listed[0]["id"] == job.id and listed[0]["status"] == "running"
        state = client.get(f"/api/jobs/{job.id}").json()
        assert state["stage"] == "extracting" and state["progress"] == 17
        assert state["current"] == 2 and state["total"] == 9
        assert client.get(f"/api/jobs/{job.id}/logs").status_code == 200
    finally:
        web.manager = original

    restarted = web.JobManager(tmp_path)
    restored = restarted.get(job.id)
    assert restored.status == "failed"
    assert "服务重启" in restored.error
    assert any("服务重启" in entry["message"] for entry in restored.logs)
    assert "服务重启" in (job.output_dir / "process.log").read_text(encoding="utf-8")


def test_model_wait_heartbeat_reports_still_running(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "MODEL_HEARTBEAT_SECONDS", 0.02)
    job = web.JobManager(tmp_path).create("source.txt", b"sample",
                                         web.RunConfig(mode="build"), "", start=False)
    release = threading.Event()

    class SlowModel:
        def complete(self, system, user, temperature):
            release.wait(1)
            return {"ok": True}

    result = []
    worker = threading.Thread(target=lambda: result.append(
        web.CheckedModel(SlowModel(), job).complete("system", "user", 0.1)))
    worker.start()
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline and not any("模型请求仍在进行" in item["message"] for item in job.logs):
        time.sleep(0.01)
    release.set()
    worker.join(timeout=1)
    assert result == [{"ok": True}]
    assert any("模型请求仍在进行" in item["message"] for item in job.logs)


def test_cancel_running_job_stops_model_and_preserves_status(tmp_path):
    entered = threading.Event()
    release = threading.Event()

    class BlockingModel:
        def complete(self, system, user, temperature):
            entered.set()
            release.wait(3)
            return {"entities": [], "relations": []}

        def cancel(self):
            release.set()

    manager = web.JobManager(tmp_path)
    manager._model = lambda *args: web.CheckedModel(BlockingModel(), args[-1])
    job = manager.create("source.txt", b"sample", web.RunConfig(mode="build"),
                         "", start=True)
    assert entered.wait(3)
    original = web.manager
    web.manager = manager
    try:
        response = TestClient(web.app).post(f"/api/jobs/{job.id}/cancel")
        assert response.status_code == 202
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and job.status != "cancelled":
            time.sleep(0.01)
        assert job.status == "cancelled"
        assert TestClient(web.app).get(f"/api/jobs/{job.id}").json()["status"] == "cancelled"
        assert TestClient(web.app).post(f"/api/jobs/{job.id}/cancel").status_code == 409
        assert "任务已中断" in (job.output_dir / "process.log").read_text(encoding="utf-8")
        assert not (job.output_dir / "graph.json").exists()
        assert web.JobManager(tmp_path).get(job.id).status == "cancelled"
    finally:
        web.manager = original
        release.set()
        manager.executor.shutdown(wait=True)


def test_cancel_queued_job_does_not_start(tmp_path):
    release = threading.Event()
    manager = web.JobManager(tmp_path, max_workers=1)
    blocker = manager.executor.submit(release.wait, 3)
    job = manager.create("source.txt", b"sample", web.RunConfig(mode="build"),
                         "", start=True)
    try:
        manager.cancel(job)
        assert job.status == "cancelled" and job.future.cancelled()
        assert not (job.output_dir / "chunks.json").exists()
    finally:
        release.set()
        blocker.result(timeout=3)
        manager.executor.shutdown(wait=True)


def test_web_rejects_invalid_endpoint_and_chunk_window():
    try:
        web.RunConfig(api_url="https://example.com/compatible-mode/v1")
        assert False
    except ValueError:
        pass


def test_neo4j_build_and_generate_are_independent(tmp_path):
    from milkg.graph import MilitaryGraph

    class Store:
        database = "neo4j"
        graphs = {}

        def create_graph(self, name):
            self.graphs["graph-1"] = MilitaryGraph()
            return "graph-1"

        def load_graph(self, graph_id):
            return MilitaryGraph.from_dict(self.graphs[graph_id].to_dict())

        def save_graph(self, graph_id, graph, previous, documents):
            self.graphs[graph_id] = MilitaryGraph.from_dict(graph.to_dict())
            return {"changed_nodes": graph.graph.number_of_nodes(),
                    "changed_edges": graph.graph.number_of_edges()}

        def close(self):
            pass

    class FakeModel:
        def complete(self, system, user, temperature):
            if "实体关系抽取器" in system:
                return {"entities": [{"name": "甲平台", "type": "Platform/Carrier"},
                                     {"name": "甲武器", "type": "Weapon System"}],
                        "relations": [{"source": "甲平台", "target": "甲武器", "type": "Equip-Carry",
                                       "evidence": "甲平台搭载甲武器"}]}
            payload = json.loads(user.split("子图：\n", 1)[1].split("\n可用于选项", 1)[0])
            source = next(n["id"] for n in payload["nodes"] if n["name"] == "甲平台")
            answer = next(n["id"] for n in payload["nodes"] if n["name"] == "甲武器")
            return {"question": "甲平台搭载的武器是____。", "options": [], "answer": "甲武器",
                    "explanation": "甲平台搭载甲武器。",
                    "supporting_facts": [payload["facts"][0]["id"]],
                    "referenced_entities": [source, answer], "answer_entities": [answer]}

    manager = web.JobManager(tmp_path)
    store = Store()
    manager._neo4j = lambda config, password: store
    manager._model = lambda *args: FakeModel()
    build = web.RunConfig(mode="build", storage="neo4j", materialize_specs=False)
    build_job = manager.create("a.txt", "甲平台搭载甲武器。".encode(), build,
                               "key", start=False, neo4j_password="secret")
    manager.run(build_job, "key", "secret")
    assert build_job.status == "completed" and build_job.graph_id == "graph-1"
    assert not (build_job.output_dir / "sft_alpaca.json").exists()
    assert "secret" not in (build_job.output_dir / "config.json").read_text(encoding="utf-8")

    generate = web.RunConfig(mode="generate", storage="neo4j", graph_id="graph-1",
                             question_types=["fill_blank"], include_atomic=True, max_atomic=1)
    generate_job = manager.create("已有图谱", None, generate, "key", start=False)
    manager.run(generate_job, "key")
    assert generate_job.status == "completed" and generate_job.source_path is None
    assert generate_job.graph_nodes == 2 and generate_job.graph_edges == 1
    assert generate_job.items[0]["answer"] == "甲武器"
    restored = web.JobManager(tmp_path).get(generate_job.id)
    assert restored.status == "completed" and restored.source_path is None


def test_web_requires_neo4j_graph_for_standalone_generation():
    for kwargs in ({"mode": "generate", "storage": "json"},
                   {"mode": "generate", "storage": "neo4j"},
                   {"mode": "build", "storage": "neo4j", "graph_action": "extend"}):
        try:
            web.RunConfig(**kwargs)
            assert False, kwargs
        except ValueError:
            pass


def test_web_local_ollama_accepts_empty_key_and_omits_provider_options(tmp_path, monkeypatch):
    monkeypatch.delenv("ALIYUN_API_KEY", raising=False)
    manager = web.JobManager(tmp_path)
    manager.executor.submit = lambda *args: None
    config = web.RunConfig(mode="build", api_url="http://127.0.0.1:11434/v1")
    original = web.manager
    web.manager = manager
    try:
        client = TestClient(web.app)
        response = client.post("/api/jobs", data={"config": config.model_dump_json()},
                               files={"file": ("source.txt", b"sample text")})
        assert response.status_code == 202
        job = manager.get(response.json()["id"])
        model = manager._model(config, "qwen3.5:4b", "", job).client
        assert model.api_key == "" and model.timeout == 300
        assert model.enable_thinking is None and model.json_mode
        assert model.reasoning_effort == "none" and model.max_tokens == 4096
        cloud = web.RunConfig(mode="build")
        rejected = client.post("/api/jobs", data={"config": cloud.model_dump_json()},
                               files={"file": ("source.txt", b"sample text")})
        assert rejected.status_code == 400
    finally:
        web.manager = original


def test_web_api_lists_graphs_and_accepts_generate_without_file(tmp_path):
    manager = web.JobManager(tmp_path)
    manager.executor.submit = lambda *args: None

    class ListingStore:
        def __init__(self, uri, user, password, database):
            assert (uri, user, password, database) == ("bolt://127.0.0.1:7687", "neo4j", "secret", "neo4j")

        def list_graphs(self):
            return [{"id": "graph-1", "name": "资料库", "node_count": 2, "edge_count": 1}]

        def migrate_graph(self, graph_id):
            assert graph_id == "graph-1"
            return {"graph_id": graph_id, "tagged_nodes": 2, "converted_relations": 1,
                    "node_count": 2, "edge_count": 1}

        def close(self):
            pass

    original_manager, original_store = web.manager, web.Neo4jGraphStore
    web.manager, web.Neo4jGraphStore = manager, ListingStore
    try:
        client = TestClient(web.app)
        listed = client.post("/api/graphs/list", json={"password": "secret"})
        assert listed.status_code == 200 and listed.json()["graphs"][0]["id"] == "graph-1"
        migrated = client.post("/api/graphs/migrate", json={"password": "secret", "graph_id": "graph-1"})
        assert migrated.status_code == 200 and migrated.json()["converted_relations"] == 1
        config = web.RunConfig(mode="generate", storage="neo4j", graph_id="graph-1")
        created = client.post("/api/jobs", data={"config": config.model_dump_json(),
                                                 "api_key": "qwen-key", "neo4j_password": "secret"})
        assert created.status_code == 202
        job = manager.get(created.json()["id"])
        assert job.source_path is None and job.config.graph_id == "graph-1"
        assert "secret" not in (job.output_dir / "config.json").read_text(encoding="utf-8")
    finally:
        web.manager, web.Neo4jGraphStore = original_manager, original_store
    try:
        web.RunConfig(max_chars=400, overlap=400)
        assert False
    except ValueError:
        pass


def test_web_batch_upload_builds_one_graph_and_restores_sources(tmp_path, monkeypatch):
    manager = web.JobManager(tmp_path)
    manager.executor.submit = lambda *args: None
    manager._model = lambda *args: object()

    def fake_extract(chunk, *args, **kwargs):
        subject = "甲平台" if "甲" in chunk.text else "乙平台"
        weapon = "甲武器" if "甲" in chunk.text else "乙武器"
        return {"chunk_id": chunk.id,
                "entities": [{"name": subject, "type": "Platform/Carrier", "chunk_id": chunk.id,
                              "confidence": 1.0, "attributes": {}},
                             {"name": weapon, "type": "Weapon System", "chunk_id": chunk.id,
                              "confidence": 1.0, "attributes": {}}],
                "relations": [{"source": subject, "target": weapon, "type": "Equip-Carry",
                               "evidence": chunk.text, "chunk_id": chunk.id, "confidence": 1.0}],
                "diagnostics": {}}

    monkeypatch.setattr(web, "extract_chunk", fake_extract)
    monkeypatch.setenv("ALIYUN_API_KEY", "batch-key")
    original = web.manager
    web.manager = manager
    try:
        config = web.RunConfig(mode="build", materialize_specs=False)
        client = TestClient(web.app)
        response = client.post("/api/jobs", data={"config": config.model_dump_json()},
                               files=[("files", ("docs/part-a.txt", "甲平台搭载甲武器。".encode())),
                                      ("files", ("part-b.txt", "乙平台搭载乙武器。".encode()))])
        assert response.status_code == 202
        job = manager.get(response.json()["id"])
        assert job.snapshot()["file_count"] == 2
        assert job.snapshot()["sources"] == ["docs/part-a.txt", "part-b.txt"]
        manager.run(job, "batch-key")
        assert job.status == "completed" and job.graph_nodes == 4 and job.graph_edges == 2
        chunks = json.loads((job.output_dir / "chunks.json").read_text(encoding="utf-8"))
        assert {row["source"] for row in chunks} == {"docs/part-a.txt", "part-b.txt"}
        restored = web.JobManager(tmp_path).get(job.id)
        assert restored.snapshot()["sources"] == ["docs/part-a.txt", "part-b.txt"]
        assert len(restored.source_paths) == 2
    finally:
        web.manager = original


def test_web_model_list_uses_aliyun_native_and_ollama_compatible_endpoints(monkeypatch):
    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request.full_url, request.get_header("Authorization"), timeout))
        if "api/v1/models" in request.full_url:
            payload = {"output": {"total": 2, "models": [
                {"model": "qwen3.5-flash"}, {"model": "qwen3.5-plus"}]}}
        else:
            payload = {"data": [{"id": "qwen3.5:4b"}, {"id": "qwen3.5:9b"}]}
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(web.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("ALIYUN_API_KEY", "env-key")
    client = TestClient(web.app)
    cloud = client.post("/api/models", json={
        "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"})
    assert cloud.status_code == 200
    assert cloud.json()["models"] == ["qwen3.5-flash", "qwen3.5-plus"]
    assert requests[0][0].startswith("https://dashscope.aliyuncs.com/api/v1/models?")
    assert requests[0][1] == "Bearer env-key"
    local = client.post("/api/models", json={"api_url": "http://127.0.0.1:11434/v1"})
    assert local.status_code == 200
    assert local.json()["models"] == ["qwen3.5:4b", "qwen3.5:9b"]
    assert requests[1] == ("http://127.0.0.1:11434/v1/models", None, 15)


def test_failed_extraction_can_resume_without_repeating_completed_chunks(tmp_path, monkeypatch):
    manager = web.JobManager(tmp_path)
    manager._model = lambda *args: object()
    calls = []
    fail_once = True

    def fake_extract(chunk, *args, **kwargs):
        nonlocal fail_once
        calls.append(chunk.id)
        if "第二" in chunk.text and fail_once:
            fail_once = False
            raise RuntimeError("prediction aborted, token repeat limit reached")
        return {"chunk_id": chunk.id, "entities": [], "relations": [], "diagnostics": {}}

    monkeypatch.setattr(web, "extract_chunk", fake_extract)
    config = web.RunConfig(mode="build", api_url="http://127.0.0.1:11434/v1",
                           materialize_specs=False)
    job = manager.create("2 份文档", [("first.txt", "第一份。".encode()),
                                     ("second.txt", "第二份。".encode())],
                         config, "", start=False)
    manager.run(job, "")
    assert job.status == "failed" and job.snapshot()["resumable"]
    assert len(json.loads((job.output_dir / "extractions.json").read_text(encoding="utf-8"))) == 1
    saved_config = json.loads((job.output_dir / "config.json").read_text(encoding="utf-8"))
    saved_config.pop("local_thinking")
    saved_config.pop("local_max_tokens")
    (job.output_dir / "config.json").write_text(json.dumps(saved_config), encoding="utf-8")
    restored = web.JobManager(tmp_path).get(job.id)
    assert restored.snapshot()["resumable"]
    resumed_manager = web.JobManager(tmp_path)
    resumed_manager._model = lambda *args: object()
    resumed = resumed_manager.get(job.id)
    scheduled = []
    resumed_manager.executor.submit = lambda *args: scheduled.append(args)
    original = web.manager
    web.manager = resumed_manager
    try:
        response = TestClient(web.app).post(f"/api/jobs/{job.id}/resume",
                                            data={"local_max_tokens": "3072",
                                                  "local_thinking": "true"})
        assert response.status_code == 202
        assert resumed.config.local_max_tokens == 3072 and resumed.config.local_thinking
        assert len(scheduled) == 1
        scheduled[0][0](*scheduled[0][1:])
    finally:
        web.manager = original
    assert resumed.status == "completed"
    assert len(calls) == 3 and calls[0] != calls[1] == calls[2]
    assert not resumed.snapshot()["resumable"]


def test_local_model_sends_bounded_output_and_disables_thinking(monkeypatch):
    captured = []

    def fake_urlopen(request, timeout):
        captured.append(json.loads(request.data))
        return io.BytesIO(json.dumps({"choices": [{"finish_reason": "stop",
                                                   "message": {"content": "{}"}}]}).encode())

    monkeypatch.setattr(web.urllib.request, "urlopen", fake_urlopen)
    model = web.OpenAICompatibleModel("http://127.0.0.1:11434/v1", "qwen3.8:27b",
                                      reasoning_effort="none", max_tokens=4096)
    assert model.complete("system", "user", 0.1) == {}
    assert captured[0]["reasoning_effort"] == "none"
    assert captured[0]["max_tokens"] == 4096


def test_cancel_closes_active_local_model_connection(monkeypatch):
    entered = threading.Event()
    closed = threading.Event()

    class Socket:
        def shutdown(self, how):
            closed.set()

    class Connection:
        def __init__(self, *args, **kwargs):
            self.sock = Socket()

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            entered.set()
            closed.wait(3)
            raise OSError("connection closed")

        def close(self):
            closed.set()

    monkeypatch.setattr(llm.http.client, "HTTPConnection", Connection)
    event = threading.Event()
    model = web.OpenAICompatibleModel("http://127.0.0.1:11434/v1", "qwen3.8:27b",
                                      cancel_event=event)
    outcome = []
    worker = threading.Thread(target=lambda: _capture_cancel(model, outcome))
    worker.start()
    assert entered.wait(3)
    model.cancel()
    worker.join(3)
    assert not worker.is_alive() and outcome == [RequestCancelled]


def _capture_cancel(model, outcome):
    try:
        model.complete("system", "user", 0.1)
    except Exception as exc:
        outcome.append(type(exc))
