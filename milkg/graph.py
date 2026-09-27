"""Entity alignment and evidence-preserving NetworkX knowledge graph."""

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import networkx as nx

from .ontology import ENTITY_TYPES, valid_relation


def normalize_name(name: str) -> str:
    return re.sub(r"[\s\-‐‑–—_·.()（）]+", "", name).casefold()


class MilitaryGraph:
    def __init__(self, synonyms: dict[str, str] | None = None, fuzzy_threshold: float = 0.9):
        self.graph = nx.MultiDiGraph()
        self.synonyms = {normalize_name(k): normalize_name(v) for k, v in (synonyms or {}).items()}
        self.fuzzy_threshold = fuzzy_threshold
        self.alias_index: dict[tuple[str, str], str] = {}

    def _canonical(self, name: str) -> str:
        key = normalize_name(name)
        visited = set()
        while key in self.synonyms and key not in visited:
            visited.add(key)
            key = self.synonyms[key]
        return key

    def _align(self, entity: dict) -> str | None:
        kind = entity["type"]
        names = [entity["name"], *entity.get("aliases", [])]
        for name in names:
            node = self.alias_index.get((kind, self._canonical(name)))
            if node:
                return node
        key = self._canonical(entity["name"])
        # Restrict fuzzy matches to sufficiently long names of the same type.
        # Attribute conflicts veto the match to avoid merging different variants.
        if len(key) >= 5:
            for node, data in self.graph.nodes(data=True):
                candidate = self._canonical(data["name"])
                if data["type"] != kind or len(candidate) < 5:
                    continue
                if SequenceMatcher(None, key, candidate).ratio() < self.fuzzy_threshold:
                    continue
                attrs = entity.get("attributes", {})
                if any(k in data["attributes"] and data["attributes"][k]["value"] != v
                       and data["attributes"][k]["confidence"] >= entity.get("confidence", 1)
                       for k, v in attrs.items()):
                    continue
                return node
        return None

    def add_entity(self, entity: dict) -> str:
        if entity["type"] not in ENTITY_TYPES:
            raise ValueError(f"Unknown entity type: {entity['type']}")
        node = self._align(entity)
        if node is None:
            node = f"n{self.graph.number_of_nodes() + 1:06d}"
            self.graph.add_node(node, name=entity["name"], type=entity["type"],
                                aliases=[], attributes={}, confidence=0.0, chunks=[])
        data = self.graph.nodes[node]
        data["confidence"] = max(data["confidence"], float(entity.get("confidence", 1)))
        for name in [entity["name"], *entity.get("aliases", [])]:
            self.alias_index[(entity["type"], self._canonical(name))] = node
            if name != data["name"] and name not in data["aliases"]:
                data["aliases"].append(name)
        chunk_id = entity.get("chunk_id")
        if chunk_id and chunk_id not in data["chunks"]:
            data["chunks"].append(chunk_id)
        for key, value in entity.get("attributes", {}).items():
            old = data["attributes"].get(key)
            confidence = float(entity.get("confidence", 1))
            if old is None or confidence > old["confidence"]:
                data["attributes"][key] = {"value": value, "confidence": confidence, "chunk_id": chunk_id}
        return node

    def add_extraction(self, extraction: dict) -> None:
        local: dict[str, str] = {}
        for entity in extraction.get("entities", []):
            local[entity["name"]] = self.add_entity(entity)
        for relation in extraction.get("relations", []):
            source, target = local.get(relation["source"]), local.get(relation["target"])
            if not source or not target or source == target:
                continue
            kind = relation["type"]
            if not valid_relation(kind, self.graph.nodes[source]["type"], self.graph.nodes[target]["type"]):
                continue
            # A typed directed pair is one KG fact with all source occurrences.
            existing = next((key for key, data in self.graph.get_edge_data(source, target, default={}).items()
                             if data["type"] == kind), None)
            evidence = {"text": relation["evidence"], "chunk_id": relation.get("chunk_id")}
            if existing is None:
                edge_id = f"e{self.graph.number_of_edges() + 1:06d}"
                self.graph.add_edge(source, target, key=edge_id, id=edge_id, type=kind,
                                    confidence=float(relation["confidence"]), evidence=[evidence],
                                    derived_from_attribute=bool(relation.get("derived_from_attribute", False)))
            else:
                data = self.graph[source][target][existing]
                data["confidence"] = max(data["confidence"], float(relation["confidence"]))
                if evidence not in data["evidence"]:
                    data["evidence"].append(evidence)

    def edges(self, kind: str | None = None):
        for source, target, key, data in self.graph.edges(keys=True, data=True):
            if kind is None or data["type"] == kind:
                yield source, target, key, data

    def edge(self, edge_id: str) -> tuple[str, str, dict] | None:
        for source, target, key, data in self.edges():
            if key == edge_id:
                return source, target, data
        return None

    def to_dict(self) -> dict:
        return {"nodes": [{"id": node, **data} for node, data in self.graph.nodes(data=True)],
                "edges": [{"source": s, "target": t, **data} for s, t, _, data in self.edges()]}

    @classmethod
    def from_dict(cls, raw: dict, synonyms: dict[str, str] | None = None) -> "MilitaryGraph":
        result = cls(synonyms)
        for item in raw.get("nodes", []):
            data = dict(item)
            node = data.pop("id")
            result.graph.add_node(node, **data)
            for name in [data["name"], *data.get("aliases", [])]:
                result.alias_index[(data["type"], result._canonical(name))] = node
        for item in raw.get("edges", []):
            data = dict(item)
            source, target, edge_id = data.pop("source"), data.pop("target"), data["id"]
            result.graph.add_edge(source, target, key=edge_id, **data)
        return result

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, synonyms: dict[str, str] | None = None) -> "MilitaryGraph":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")), synonyms)
