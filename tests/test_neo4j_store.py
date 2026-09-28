"""Exercise the persistence contract without requiring a running Neo4j server."""

import json

import pytest

from milkg.graph import MilitaryGraph
from milkg.neo4j_store import Neo4jGraphStore


class MemoryDriver:
    def __init__(self):
        self.graphs = {}
        self.nodes = {}
        self.edges = {}
        self.documents = {}
        self.links = set()

    def verify_connectivity(self):
        pass

    def close(self):
        pass

    def session(self, database):
        driver = self

        class Result:
            def __init__(self, rows):
                self.rows = rows

            def single(self):
                return self.rows[0] if self.rows else None

            def consume(self):
                return None

        class Transaction:
            def run(self, query, **params):
                rows, _, _ = driver.execute_query(query, database_=database, **params)
                return Result(rows)

        class Session:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def execute_write(self, fn, *args):
                return fn(Transaction(), *args)

        return Session()

    def execute_query(self, query, database_, **params):
        rows = params.get("rows", [])
        if query.startswith(("CREATE CONSTRAINT", "CREATE INDEX")):
            result = []
        elif query.startswith("CREATE (g:MilKGGraph"):
            self.graphs[params["id"]] = dict(id=params["id"], name=params["name"],
                                               created_at=params["now"], updated_at=params["now"],
                                               node_count=0, edge_count=0, version=0)
            result = []
        elif "RETURN g.id AS id" in query:
            if "{id: $id}" in query:
                result = [self.graphs[params["id"]]] if params["id"] in self.graphs else []
            else:
                result = list(self.graphs.values())
        elif "RETURN properties(n) AS data" in query:
            result = [{"data": n} for n in self.nodes.values()
                      if n["graph_id"] == params["graph_id"]]
        elif "RETURN a.id AS source" in query:
            result = [{"source": self.nodes[e["source_uid"]]["id"],
                       "target": self.nodes[e["target_uid"]]["id"], "data": e}
                      for e in self.edges.values() if e["graph_id"] == params["graph_id"]]
        elif "MERGE (n:MilKGEntity" in query:
            for row in rows:
                self.nodes[row["uid"]] = {**row, "graph_id": params["graph_id"]}
            result = []
        elif "MERGE (a)-[r:MILKG_RELATION" in query:
            for row in rows:
                assert row["source_uid"] in self.nodes and row["target_uid"] in self.nodes
                self.edges[row["uid"]] = {**row, "graph_id": params["graph_id"]}
            result = []
        elif "MERGE (d:MilKGDocument" in query:
            for row in rows:
                self.documents[row["uid"]] = {**row, "graph_id": params["graph_id"]}
            result = []
        elif "MERGE (n)-[:MENTIONED_IN]" in query:
            for row in rows:
                self.links.add((row["entity_uid"], row["document_uid"]))
            result = []
        elif "SET g._milkg_lock" in query:
            graph = self.graphs.get(params["id"])
            result = [{"version": graph["version"]}] if graph else []
        elif "SET g.updated_at" in query:
            self.graphs[params["id"]].update(updated_at=params["now"],
                                               node_count=params["nodes"], edge_count=params["edges"],
                                               version=params["version"])
            result = []
        else:
            raise AssertionError(query)
        return result, None, None


def _extract(platform, weapon, chunk_id):
    return {"entities": [{"name": platform, "type": "Platform/Carrier", "chunk_id": chunk_id,
                          "confidence": 0.9},
                         {"name": weapon, "type": "Weapon System", "chunk_id": chunk_id,
                          "confidence": 0.9}],
            "relations": [{"source": platform, "target": weapon, "type": "Equip-Carry",
                           "evidence": f"{platform}搭载{weapon}", "chunk_id": chunk_id,
                           "confidence": 0.9}]}


def test_neo4j_graph_namespace_extension_and_document_provenance():
    driver = MemoryDriver()
    store = Neo4jGraphStore("bolt://local", "neo4j", "secret", driver=driver)
    graph_id = store.create_graph("轻武器")
    kg = MilitaryGraph()
    kg.add_extraction(_extract("甲平台", "甲武器", "doc1:0000"))
    first = store.save_graph(graph_id, kg, documents=[{"id": "doc1", "name": "a.txt", "source": "a.txt"}])
    assert first["changed_nodes"] == 2 and first["changed_edges"] == 1
    assert len(driver.documents) == 1 and len(driver.links) == 2

    loaded = store.load_graph(graph_id)
    previous = MilitaryGraph.from_dict(loaded.to_dict())
    loaded.add_extraction(_extract("甲平台", "乙武器", "doc2:0000"))
    second = store.save_graph(graph_id, loaded, previous,
                              [{"id": "doc2", "name": "b.txt", "source": "b.txt"}])
    assert second["node_count"] == 3 and second["edge_count"] == 2
    assert second["changed_nodes"] == 2 and second["changed_edges"] == 1
    assert len(driver.links) == 4
    assert store.save_graph(graph_id, loaded, MilitaryGraph.from_dict(loaded.to_dict()))["changed_nodes"] == 0

    other_id = store.create_graph("独立图谱")
    other = MilitaryGraph()
    other.add_extraction(_extract("甲平台", "丙武器", "doc3:0000"))
    store.save_graph(other_id, other)
    assert store.load_graph(graph_id).graph.number_of_edges() == 2
    assert store.load_graph(other_id).graph.number_of_edges() == 1
    assert len(store.list_graphs()) == 2


def test_cli_build_extend_and_traverse_neo4j_without_llm(tmp_path, monkeypatch):
    from milkg import cli

    driver = MemoryDriver()
    store = Neo4jGraphStore("bolt://local", "neo4j", "secret", driver=driver)
    monkeypatch.setattr(cli, "_neo4j", lambda args: store)
    first_file, second_file = tmp_path / "first.json", tmp_path / "second.json"
    first_file.write_text(json.dumps([_extract("甲平台", "甲武器", "doc1:0000")],
                                     ensure_ascii=False), encoding="utf-8")
    second_file.write_text(json.dumps([_extract("甲平台", "乙武器", "doc2:0000")],
                                      ensure_ascii=False), encoding="utf-8")
    out1, out2, out3 = (tmp_path / name for name in ("build1", "build2", "traverse"))
    assert cli.main(["build", "--store", "neo4j", "--extractions", str(first_file),
                     "--output", str(out1)]) == 0
    graph_id = json.loads((out1 / "graph_ref.json").read_text(encoding="utf-8"))["graph_id"]
    assert cli.main(["build", "--store", "neo4j", "--graph-action", "extend", "--graph-id", graph_id,
                     "--extractions", str(second_file), "--output", str(out2)]) == 0
    assert cli.main(["traverse", "--store", "neo4j", "--graph-id", graph_id,
                     "--output", str(out3)]) == 0
    assert cli.main(["graphs", "--store", "neo4j", "--output", str(tmp_path / "catalog")]) == 0
    assert store.load_graph(graph_id).graph.number_of_nodes() == 3
    assert store.load_graph(graph_id).graph.number_of_edges() == 2
    assert (out3 / "subgraphs.json").exists()


def test_concurrent_extension_rejects_stale_graph():
    driver = MemoryDriver()
    first = Neo4jGraphStore("bolt://local", "neo4j", "secret", driver=driver)
    second = Neo4jGraphStore("bolt://local", "neo4j", "secret", driver=driver)
    graph_id = first.create_graph("共享图谱")
    base = MilitaryGraph()
    base.add_extraction(_extract("甲平台", "甲武器", "doc1:0000"))
    first.save_graph(graph_id, base)
    version_a, version_b = first.load_graph(graph_id), second.load_graph(graph_id)
    old_a, old_b = (MilitaryGraph.from_dict(g.to_dict()) for g in (version_a, version_b))
    version_a.add_extraction(_extract("甲平台", "乙武器", "doc2:0000"))
    version_b.add_extraction(_extract("甲平台", "丙武器", "doc3:0000"))
    first.save_graph(graph_id, version_a, old_a)
    with pytest.raises(RuntimeError, match="其他任务更新"):
        second.save_graph(graph_id, version_b, old_b)
    assert first.load_graph(graph_id).graph.number_of_edges() == 2
