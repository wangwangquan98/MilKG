"""Document-specific human review for output/test_input_qwen/qa.json.

Keeps the original model output as qa_uncurated.json and records every edit.
"""

import json
import shutil
from pathlib import Path

from milkg.graph import MilitaryGraph
from milkg.qa import export_sft, validate_qa
from milkg.traversal import Subgraph


root = Path("output/test_input_qwen")
qa_path = root / "qa.json"
original = root / "qa_uncurated.json"
if not original.exists():
    shutil.copyfile(qa_path, original)
items = json.loads(original.read_text(encoding="utf-8"))
kg = MilitaryGraph.load(root / "graph_augmented.json")
edits = []
supplement_path = root / "multiple_choice_candidates.json"
if supplement_path.exists():
    items.extend(json.loads(supplement_path.read_text(encoding="utf-8")))

for item in items:
    question = item["question"]
    if question.startswith("根据自动方式的技术特征，大多数____"):
        item["question"] = "自卫手枪的自动方式大多数采用____和枪管短后座式。"
        item["answer"] = "枪机后座"
        item["answer_entities"] = [kg.edge("e000004")[1]]
        item["explanation"] = "原文在自卫手枪部分写明，多数采用枪机后座和枪管短后座式；事实 e000004 记录了枪机后座。"
        edits.append({"item": "fill_blank:自卫手枪", "reason": "修正题干的过度概括及截断解释"})
    elif question.startswith("根据已知事实，下列哪种武器系统被明确记录为美军的装备"):
        item["explanation"] = "事实 e000003 记录了美军与 M79 榴弹发射器的装备关系，因此选 A。"
        edits.append({"item": "single_choice:美军装备", "reason": "替换截断解释"})
    elif question.startswith("作为指挥员，若需部署一种用于毁伤低空目标"):
        item["explanation"] = "原文指出大口径机枪用于毁伤低空目标，对空有效射程为 2000 米；事实 e000018 记录了该射程，因此选 A。"
        edits.append({"item": "single_choice:大口径机枪", "reason": "替换截断解释"})
    elif question.startswith("在近距离作战决策中，若需选择一种威力介于步枪和机枪之间"):
        item["explanation"] = "原文指出冲锋枪威力介于步枪和机枪之间，并且结构简单、短小轻便；事实 e000014 记录了威力规格，因此选 A。"
        edits.append({"item": "single_choice:冲锋枪", "reason": "删除缺少来源的干扰项比较"})
    elif question.startswith("根据现有资料判断：遂发火枪"):
        item["explanation"] = "事实 e000012 记录了遂发火枪利用遂石打火机构点火发射，因此判断为正确。"
        edits.append({"item": "true_false:遂发火枪", "reason": "删除基于图谱缺失关系的推断"})
    elif item["type"] == "multiple_choice" and "大口径机枪的战术性能" in question:
        item["question"] = "依据给定知识图谱，哪些技术规格有来源证据证明属于大口径机枪？"
        item["options"] = [
            {"label": "A", "text": kg.graph.nodes["n000135"]["name"]},
            {"label": "B", "text": kg.graph.nodes["n000139"]["name"]},
            {"label": "C", "text": kg.graph.nodes["n000140"]["name"]},
            {"label": "D", "text": kg.graph.nodes["n000065"]["name"]},
        ]
        item["answer"] = ["B", "C"]
        item["answer_entities"] = ["n000139", "n000140"]
        item["referenced_entities"] = ["n000118", "n000139", "n000140", "n000135", "n000065"]
        item["explanation"] = ("事实 e000018 和 e000019 分别证明大口径机枪的对空有效射程为 2000 米、"
                               "对地有效射程为 1000~1500 米。A 和 D 在给定子图中没有对应于该武器的来源边，"
                               "因此可由来源证实的选项是 B、C。")
        edits.append({"item": "multiple_choice:大口径机枪", "reason": "统一选项类型并去除无意义的主语选项"})

for item in items:
    edges = item["subgraph_edges"]
    nodes = tuple(dict.fromkeys(n for edge_id in edges for n in kg.edge(edge_id)[:2]))
    subgraph = Subgraph(item["strategy"], nodes, tuple(edges), len(edges))
    valid, reason = validate_qa(item, kg, subgraph, item["type"], item["answer"] != "错误")
    if not valid:
        raise ValueError(f"Curated item failed validation: {reason}: {item['question']}")

qa_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
export_sft(items, root / "sft_alpaca.json", "alpaca")
supplement_stats_path = root / "multiple_choice_stats.json"
if supplement_stats_path.exists():
    main_stats_path = root / "stats_main.json"
    if not main_stats_path.exists():
        shutil.copyfile(root / "stats.json", main_stats_path)
    main_stats = json.loads(main_stats_path.read_text(encoding="utf-8"))
    supplement_stats = json.loads(supplement_stats_path.read_text(encoding="utf-8"))
    for key, value in supplement_stats.items():
        if isinstance(value, int):
            main_stats[key] = main_stats.get(key, 0) + value
    (root / "stats.json").write_text(json.dumps(main_stats, ensure_ascii=False, indent=2), encoding="utf-8")
(root / "curation_report.json").write_text(
    json.dumps({"items": len(items), "edits": edits, "validation": "all passed"},
               ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Curated {len(items)} items; {len(edits)} edited")
