"""Persistent, namespaced MilKG graphs in Neo4j.

Graph IDs are logical namespaces inside one Neo4j database. Creating a graph
never deletes another graph; extending one keeps node/edge IDs and provenance.
"""

import json
import uuid
from datetime import datetime, timezone

from .graph import MilitaryGraph
from .ontology import ENTITY_TYPES, RELATIONS


SCHEMA_VERSION = 2


def _ontology_name(name: str, allowed) -> str:
    """Quote a fixed ontology name for Cypher labels and relationship types."""
    if name not in allowed:
        raise ValueError(f"Unknown ontology type: {name}")
    return "`" + name.replace("`", "``") + "`"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def document_rows(chunks) -> list[dict]:
    """Describe source documents using the stable content-hash chunk prefix."""
    found = {}
    for chunk in chunks:
        doc_id = chunk.id.split(":", 1)[0]
        found[doc_id] = {"id": doc_id, "source": chunk.source,
                         "name": chunk.source.replace("\\", "/").rsplit("/", 1)[-1]}
    return list(found.values())


class Neo4jGraphStore:
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j",
                 driver=None):
        if driver is None:
            try:
                from neo4j import GraphDatabase
            except ImportError as exc:
                raise RuntimeError("Neo4j support requires: pip install -e '.[neo4j]'") from exc
            driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver = driver
        self.database = database
        self._versions: dict[str, int] = {}
        self._schema_versions: dict[str, int] = {}
        try:
            self.driver.verify_connectivity()
        except Exception:
            self.driver.close()
            raise

    def close(self) -> None:
        self.driver.close()

    def _query(self, query: str, **params):
        records, _, _ = self.driver.execute_query(query, database_=self.database, **params)
        return records

    def ensure_schema(self) -> None:
        for query in (
            "CREATE CONSTRAINT milkg_graph_id IF NOT EXISTS FOR (g:MilKGGraph) REQUIRE g.id IS UNIQUE",
            "CREATE CONSTRAINT milkg_entity_uid IF NOT EXISTS FOR (n:MilKGEntity) REQUIRE n.uid IS UNIQUE",
            "CREATE CONSTRAINT milkg_document_uid IF NOT EXISTS FOR (d:MilKGDocument) REQUIRE d.uid IS UNIQUE",
            "CREATE INDEX milkg_entity_graph_id IF NOT EXISTS FOR (n:MilKGEntity) ON (n.graph_id)",
            "CREATE INDEX milkg_document_graph_id IF NOT EXISTS FOR (d:MilKGDocument) ON (d.graph_id)",
        ):
            self._query(query)

    def list_graphs(self) -> list[dict]:
        rows = self._query("MATCH (g:MilKGGraph) RETURN g.id AS id, g.name AS name, "
                           "g.created_at AS created_at, g.updated_at AS updated_at, "
                           "g.node_count AS node_count, g.edge_count AS edge_count, "
                           "g.version AS version, g.schema_version AS schema_version "
                           "ORDER BY g.updated_at DESC")
        return [dict(row) for row in rows]

    def get_graph(self, graph_id: str) -> dict:
        rows = self._query("MATCH (g:MilKGGraph {id: $id}) RETURN g.id AS id, g.name AS name, "
                           "g.created_at AS created_at, g.updated_at AS updated_at, "
                           "g.node_count AS node_count, g.edge_count AS edge_count, "
                           "g.version AS version, g.schema_version AS schema_version", id=graph_id)
        if not rows:
            raise ValueError(f"Neo4j graph does not exist: {graph_id}")
        graph = dict(rows[0])
        self._versions[graph_id] = graph.get("version") or 0
        self._schema_versions[graph_id] = graph.get("schema_version") or 1
        return graph

    def create_graph(self, name: str) -> str:
        self.ensure_schema()
        graph_id = uuid.uuid4().hex
        self._query("CREATE (g:MilKGGraph {id: $id, name: $name, created_at: $now, "
                    "updated_at: $now, node_count: 0, edge_count: 0, version: 0, "
                    "schema_version: $schema_version})",
                    id=graph_id, name=name.strip() or graph_id, now=_now(),
                    schema_version=SCHEMA_VERSION)
        self._versions[graph_id] = 0
        self._schema_versions[graph_id] = SCHEMA_VERSION
        return graph_id

    def load_graph(self, graph_id: str) -> MilitaryGraph:
        self.get_graph(graph_id)
        nodes = []
        for row in self._query("MATCH (n:MilKGEntity {graph_id: $graph_id}) "
                               "RETURN properties(n) AS data", graph_id=graph_id):
            data = dict(row["data"])
            nodes.append({"id": data["id"], "name": data["name"], "type": data["entity_type"],
                          "aliases": data.get("aliases", []),
                          "attributes": json.loads(data.get("attributes_json", "{}")),
                          "confidence": data.get("confidence", 0.0),
                          "chunks": data.get("chunks", [])})
        edges = []
        for row in self._query("MATCH (a:MilKGEntity {graph_id: $graph_id})"
                               "-[r]->(b:MilKGEntity {graph_id: $graph_id}) "
                               "WHERE type(r) IN $relation_types "
                               "RETURN a.id AS source, b.id AS target, properties(r) AS data",
                               graph_id=graph_id,
                               relation_types=[*RELATIONS, "MILKG_RELATION"]):
            data = dict(row["data"])
            edges.append({"id": data["id"], "source": row["source"], "target": row["target"],
                          "type": data["relation_type"],
                          "confidence": data.get("confidence", 0.0),
                          "evidence": json.loads(data.get("evidence_json", "[]")),
                          "derived_from_attribute": data.get("derived_from_attribute", False)})
        return MilitaryGraph.from_dict({"nodes": nodes, "edges": edges})

    def save_graph(self, graph_id: str, graph: MilitaryGraph,
                   previous: MilitaryGraph | None = None,
                   documents: list[dict] | None = None) -> dict:
        if graph_id not in self._versions:
            self.get_graph(graph_id)
        current = graph.to_dict()
        before = previous.to_dict() if previous else {"nodes": [], "edges": []}
        old_nodes = {item["id"]: item for item in before["nodes"]}
        old_edges = {item["id"]: item for item in before["edges"]}
        changed_nodes = [item for item in current["nodes"] if old_nodes.get(item["id"]) != item]
        changed_edges = [item for item in current["edges"] if old_edges.get(item["id"]) != item]
        unknown_entities = {node["type"] for node in changed_nodes} - set(ENTITY_TYPES)
        unknown_relations = {edge["type"] for edge in changed_edges} - set(RELATIONS)
        if unknown_entities or unknown_relations:
            raise ValueError(f"图谱包含未定义的本体类型：{sorted(unknown_entities | unknown_relations)}")
        node_rows = [{"uid": f"{graph_id}:{n['id']}", "id": n["id"], "name": n["name"],
                      "entity_type": n["type"], "aliases": n.get("aliases", []),
                      "attributes_json": json.dumps(n.get("attributes", {}), ensure_ascii=False),
                      "confidence": n.get("confidence", 0.0), "chunks": n.get("chunks", [])}
                     for n in changed_nodes]
        edge_rows = [{"uid": f"{graph_id}:{e['id']}", "id": e["id"],
                      "source_uid": f"{graph_id}:{e['source']}",
                      "target_uid": f"{graph_id}:{e['target']}",
                      "relation_type": e["type"], "confidence": e.get("confidence", 0.0),
                      "evidence_json": json.dumps(e.get("evidence", []), ensure_ascii=False),
                      "document_ids": sorted({ev.get("chunk_id", "").split(":", 1)[0]
                                              for ev in e.get("evidence", []) if ev.get("chunk_id")}),
                      "derived_from_attribute": e.get("derived_from_attribute", False)}
                     for e in changed_edges]
        docs = documents or []
        doc_rows = [{**d, "uid": f"{graph_id}:{d['id']}"} for d in docs]
        links = [{"entity_uid": f"{graph_id}:{n['id']}",
                  "document_uid": f"{graph_id}:{doc['id']}"}
                 for n in changed_nodes for doc in docs
                 if any(chunk.startswith(doc["id"] + ":") for chunk in n.get("chunks", []))]
        with self.driver.session(database=self.database) as session:
            version, _, _ = session.execute_write(self._write_graph, graph_id,
                                                  self._versions[graph_id], node_rows,
                                                  edge_rows, doc_rows, links,
                                                  len(current["nodes"]), len(current["edges"]),
                                                  self._schema_versions[graph_id] < SCHEMA_VERSION)
        self._versions[graph_id] = version
        self._schema_versions[graph_id] = SCHEMA_VERSION
        return {"changed_nodes": len(changed_nodes), "changed_edges": len(changed_edges),
                "node_count": len(current["nodes"]), "edge_count": len(current["edges"])}

    def migrate_graph(self, graph_id: str) -> dict:
        """Upgrade an existing logical graph without rebuilding or calling an LLM."""
        metadata = self.get_graph(graph_id)
        with self.driver.session(database=self.database) as session:
            version, tagged, converted = session.execute_write(
                self._write_graph, graph_id, self._versions[graph_id], [], [], [], [],
                metadata["node_count"], metadata["edge_count"], True)
        self._versions[graph_id] = version
        self._schema_versions[graph_id] = SCHEMA_VERSION
        return {"graph_id": graph_id, "tagged_nodes": tagged,
                "converted_relations": converted,
                "node_count": metadata["node_count"], "edge_count": metadata["edge_count"]}

    @staticmethod
    def _migrate_ontology(tx, graph_id: str) -> tuple[int, int]:
        tagged = converted = 0
        for kind in ENTITY_TYPES:
            label = _ontology_name(kind, ENTITY_TYPES)
            record = tx.run(
                "MATCH (n:MilKGEntity {graph_id: $graph_id, entity_type: $kind}) "
                f"WHERE NOT $kind IN labels(n) SET n:{label} "
                "RETURN count(n) AS changed", graph_id=graph_id, kind=kind).single()
            tagged += record["changed"]
        for kind in RELATIONS:
            relation_type = _ontology_name(kind, RELATIONS)
            record = tx.run(
                "MATCH (a:MilKGEntity {graph_id: $graph_id})"
                "-[old:MILKG_RELATION {relation_type: $kind}]->"
                "(b:MilKGEntity {graph_id: $graph_id}) "
                f"MERGE (a)-[r:{relation_type} {{uid: old.uid}}]->(b) "
                "SET r = properties(old) DELETE old "
                "RETURN count(*) AS changed", graph_id=graph_id, kind=kind).single()
            converted += record["changed"]
        return tagged, converted

    @staticmethod
    def _write_graph(tx, graph_id: str, expected_version: int, node_rows: list[dict],
                     edge_rows: list[dict], doc_rows: list[dict], links: list[dict],
                     node_count: int, edge_count: int,
                     migrate_legacy: bool = False) -> tuple[int, int, int]:
        # Take the catalog-node write lock before reading its version. Neo4j's
        # default read-committed isolation otherwise allows two stale writers.
        record = tx.run("MATCH (g:MilKGGraph {id: $id}) "
                        "SET g._milkg_lock = true REMOVE g._milkg_lock "
                        "WITH g RETURN g.version AS version", id=graph_id).single()
        if record is None or record["version"] != expected_version:
            raise RuntimeError("图谱已被其他任务更新，请重新读取图谱后重试本批文档")
        tagged, converted = Neo4jGraphStore._migrate_ontology(tx, graph_id) if migrate_legacy else (0, 0)
        for kind in ENTITY_TYPES:
            rows = [row for row in node_rows if row["entity_type"] == kind]
            label = _ontology_name(kind, ENTITY_TYPES)
            for batch in _batches(rows):
                tx.run("UNWIND $rows AS row MERGE (n:MilKGEntity {uid: row.uid}) "
                       f"SET n += row, n.graph_id = $graph_id SET n:{label}",
                       rows=batch, graph_id=graph_id).consume()
        for kind in RELATIONS:
            rows = [row for row in edge_rows if row["relation_type"] == kind]
            relation_type = _ontology_name(kind, RELATIONS)
            for batch in _batches(rows):
                tx.run("UNWIND $rows AS row "
                       "MATCH (a:MilKGEntity {uid: row.source_uid}) "
                       "MATCH (b:MilKGEntity {uid: row.target_uid}) "
                       f"MERGE (a)-[r:{relation_type} {{uid: row.uid}}]->(b) "
                       "SET r += row, r.graph_id = $graph_id", rows=batch,
                       graph_id=graph_id).consume()
        for batch in _batches(doc_rows):
            tx.run("UNWIND $rows AS row MERGE (d:MilKGDocument {uid: row.uid}) "
                   "SET d += row, d.graph_id = $graph_id", rows=batch, graph_id=graph_id).consume()
        for batch in _batches(links):
            tx.run("UNWIND $rows AS row "
                   "MATCH (n:MilKGEntity {uid: row.entity_uid}) "
                   "MATCH (d:MilKGDocument {uid: row.document_uid}) "
                   "MERGE (n)-[:MENTIONED_IN]->(d)", rows=batch).consume()
        tx.run("MATCH (g:MilKGGraph {id: $id}) SET g.updated_at = $now, "
               "g.node_count = $nodes, g.edge_count = $edges, g.version = $version, "
               "g.schema_version = $schema_version", id=graph_id,
               version=expected_version + 1,
               now=_now(), nodes=node_count, edges=edge_count,
               schema_version=SCHEMA_VERSION).consume()
        return expected_version + 1, tagged, converted


def _batches(rows: list[dict], size: int = 200):
    for start in range(0, len(rows), size):
        yield rows[start:start + size]
