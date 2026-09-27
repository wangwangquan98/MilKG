"""Audit the generated SFT dataset against the saved source chunks and KG."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from milkg.graph import MilitaryGraph
from milkg.qa import validate_qa
from milkg.traversal import Subgraph


root = Path("output/test_input_qwen")
chunks = {item["id"]: item["text"] for item in json.loads((root / "chunks.json").read_text(encoding="utf-8"))}
kg = MilitaryGraph.load(root / "graph_augmented.json")
items = json.loads((root / "qa.json").read_text(encoding="utf-8"))
sft = json.loads((root / "sft_alpaca.json").read_text(encoding="utf-8"))
stats = json.loads((root / "stats.json").read_text(encoding="utf-8"))
source = Path("test_input.txt")

evidence_count = 0
for _, _, _, data in kg.edges():
    for evidence in data["evidence"]:
        if evidence["text"] not in chunks[evidence["chunk_id"]]:
            raise ValueError(f"Untraceable KG evidence: {data['id']}")
        evidence_count += 1

for item in items:
    edges = item["subgraph_edges"]
    nodes = tuple(dict.fromkeys(n for edge_id in edges for n in kg.edge(edge_id)[:2]))
    subgraph = Subgraph(item["strategy"], nodes, tuple(edges), len(edges))
    valid, reason = validate_qa(item, kg, subgraph, item["type"], item["answer"] != "错误")
    if not valid:
        raise ValueError(f"Invalid QA: {reason}: {item['question']}")

if len(sft) != len(items):
    raise ValueError("SFT/QA item count mismatch")
if any(not {"instruction", "input", "output"} <= set(item) for item in sft):
    raise ValueError("Invalid Alpaca record")

report = {
    "source": str(source.resolve()),
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "extract_model": "qwen3.5-flash",
    "generate_model": "qwen3.5-plus",
    "source_chunks": len(chunks),
    "graph_nodes": kg.graph.number_of_nodes(),
    "graph_edges": kg.graph.number_of_edges(),
    "source_backed_evidence_spans": evidence_count,
    "attempted_qa": stats["attempted"],
    "accepted_qa": len(items),
    "qa_types": dict(Counter(item["type"] for item in items)),
    "qa_strategies": dict(Counter(item["strategy"] for item in items)),
    "automatic_validation": "all passed",
    "human_curation": json.loads((root / "curation_report.json").read_text(encoding="utf-8")),
}
(root / "quality_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({key: report[key] for key in ("source_chunks", "graph_nodes", "graph_edges",
                                              "attempted_qa", "accepted_qa", "qa_types")},
                 ensure_ascii=False))
