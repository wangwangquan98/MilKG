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
    try:
        web.RunConfig(max_chars=400, overlap=400)
        assert False
    except ValueError:
        pass
