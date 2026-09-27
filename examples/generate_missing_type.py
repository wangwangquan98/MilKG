"""Generate a small, source-grounded multiple-choice supplement for this run."""

import json
import os
from pathlib import Path

from milkg.graph import MilitaryGraph
from milkg.llm import OpenAICompatibleModel
from milkg.qa import generate_qa, rouge_l
from milkg.traversal import Subgraph


root = Path("output/test_input_qwen")
kg = MilitaryGraph.load(root / "graph_augmented.json")
edges = ("e000018", "e000019")
nodes = tuple(dict.fromkeys(n for edge_id in edges for n in kg.edge(edge_id)[:2]))
subgraph = Subgraph("multi_hop", nodes, edges, 2)
model = OpenAICompatibleModel("https://dashscope.aliyuncs.com/compatible-mode/v1",
                            "qwen3.5-plus", os.environ["ALIYUN_API_KEY"], enable_thinking=False)
items, stats = generate_qa(kg, [subgraph], model, ("multiple_choice",), seed=13,
                           per_subgraph=3)
existing = json.loads((root / "qa.json").read_text(encoding="utf-8"))
new = [item for item in items if not any(rouge_l(item["question"], prior["question"]) >= 0.85
                                         for prior in existing)]
(root / "multiple_choice_candidates.json").write_text(json.dumps(new, ensure_ascii=False, indent=2), encoding="utf-8")
(root / "multiple_choice_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"attempted": stats["attempted"], "accepted": len(new),
                  "rejected": stats["invalid"]}, ensure_ascii=False))
