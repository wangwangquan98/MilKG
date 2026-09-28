import json

from fastapi.testclient import TestClient

from milkg import web


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


def test_web_api_lists_graphs_and_accepts_generate_without_file(tmp_path):
    manager = web.JobManager(tmp_path)
    manager.executor.submit = lambda *args: None

    class ListingStore:
        def __init__(self, uri, user, password, database):
            assert (uri, user, password, database) == ("bolt://127.0.0.1:7687", "neo4j", "secret", "neo4j")

        def list_graphs(self):
            return [{"id": "graph-1", "name": "资料库", "node_count": 2, "edge_count": 1}]

        def close(self):
            pass

    original_manager, original_store = web.manager, web.Neo4jGraphStore
    web.manager, web.Neo4jGraphStore = manager, ListingStore
    try:
        client = TestClient(web.app)
        listed = client.post("/api/graphs/list", json={"password": "secret"})
        assert listed.status_code == 200 and listed.json()["graphs"][0]["id"] == "graph-1"
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
