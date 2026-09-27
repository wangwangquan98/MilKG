import json

from milkg.documents import Chunk, chunk_text
from milkg.extraction import validate_extraction
from milkg.graph import MilitaryGraph
from milkg.qa import export_sft, generate_qa, rouge_l, validate_qa
from milkg.training import evaluate_extractions, prepare_extractor_sft
from milkg.evaluation import evaluate_qa, random_edge_baseline, traversal_metrics
from milkg.augmentation import materialize_specifications
from milkg.traversal import (Subgraph, atomic_facts, campaign_panoramas, conflict_chains,
                             entity_comparisons, equipment_chains, multi_hop_paths,
                             traverse)


def fixture_graph():
    kg = MilitaryGraph({"甲型平台": "甲平台"})
    entities = [
        ("甲武器", "Weapon System"), ("乙武器", "Weapon System"),
        ("丙武器", "Weapon System"), ("丁武器", "Weapon System"),
        ("甲平台", "Platform/Carrier"), ("乙平台", "Platform/Carrier"),
        ("甲单位", "Combat Unit"), ("甲司令部", "Command Structure"),
        ("甲行动", "Campaign/Operation"), ("乙行动", "Campaign/Operation"),
        ("甲参数", "Technical Specification"), ("乙参数", "Technical Specification"),
    ]
    relations = [
        ("甲平台", "甲武器", "Equip-Carry"),
        ("乙平台", "乙武器", "Equip-Carry"),
        ("甲单位", "甲平台", "Unit-Equip"),
        ("甲单位", "甲司令部", "Unit-Org"),
        ("甲武器", "乙武器", "Equip-Counter"),
        ("甲武器", "甲参数", "Weapon-Spec"),
        ("乙武器", "乙参数", "Weapon-Spec"),
        ("甲行动", "甲单位", "Campaign-Part"),
        ("甲行动", "乙行动", "Causal-Lead"),
    ]
    kg.add_extraction({"entities": [{"name": name, "type": kind, "confidence": 0.9,
                                      "attributes": {}, "chunk_id": "fixture"} for name, kind in entities],
                       "relations": [{"source": s, "target": t, "type": kind, "confidence": 0.9,
                                      "evidence": f"{s} {kind} {t}", "chunk_id": "fixture"}
                                     for s, t, kind in relations]})
    return kg


def test_extraction_checks_evidence_and_direction():
    chunk = Chunk("x:0", "x.txt", "甲平台搭载甲武器。甲单位操作甲平台。")
    raw = {"entities": [{"name": "甲平台", "type": "Platform/Carrier"},
                        {"name": "甲武器", "type": "Weapon System"},
                        {"name": "凭空武器", "type": "Weapon System"}],
           "relations": [{"source": "甲平台", "target": "甲武器", "type": "Equip-Carry",
                          "evidence": "甲平台搭载甲武器"},
                         {"source": "甲武器", "target": "甲平台", "type": "Equip-Carry",
                          "evidence": "甲平台搭载甲武器"}]}
    result = validate_extraction(raw, chunk)
    assert len(result["entities"]) == 2
    assert len(result["relations"]) == 1


def test_alignment_dedup_and_round_trip(tmp_path):
    kg = fixture_graph()
    before = kg.graph.number_of_nodes()
    kg.add_entity({"name": "甲型平台", "type": "Platform/Carrier", "confidence": 0.8,
                   "attributes": {"吨位": "100"}, "chunk_id": "x"})
    assert kg.graph.number_of_nodes() == before
    path = tmp_path / "graph.json"
    kg.save(path)
    loaded = MilitaryGraph.load(path)
    assert loaded.graph.number_of_edges() == kg.graph.number_of_edges()
    assert loaded.graph.number_of_nodes() == kg.graph.number_of_nodes()


def test_five_traversal_rules():
    kg = fixture_graph()
    assert any(s.depth == 3 for s in equipment_chains(kg))
    assert any(s.depth == 3 for s in conflict_chains(kg))
    assert any(s.depth == 3 for s in campaign_panoramas(kg))
    assert entity_comparisons(kg)
    assert any(s.depth == 4 for s in multi_hop_paths(kg))
    assert len({s.strategy for s in traverse(kg)}) == 5
    assert len(atomic_facts(kg)) == kg.graph.number_of_edges()


def test_qa_validation_and_export(tmp_path):
    kg = fixture_graph()
    edge = next(e for e in kg.edges("Equip-Carry") if kg.graph.nodes[e[1]]["name"] == "甲武器")
    subgraph = Subgraph("test", (edge[0], edge[1]), (edge[2],), 1)
    raw = {"question": "甲平台所搭载的武器是哪一项？", "options": [
        {"label": "A", "text": "甲武器"}, {"label": "B", "text": "乙武器"},
        {"label": "C", "text": "丙武器"}, {"label": "D", "text": "丁武器"}],
        "answer": ["A"], "explanation": "甲平台搭载甲武器。",
        "supporting_facts": [edge[2]], "referenced_entities": [edge[0], edge[1]],
        "answer_entities": [edge[1]]}
    assert validate_qa(raw, kg, subgraph, "single_choice")[0]
    bad = {**raw, "options": [raw["options"][0]] * 4}
    assert not validate_qa(bad, kg, subgraph, "single_choice")[0]
    item = {**raw, "type": "single_choice", "style": "formal_exam",
            "difficulty": "easy", "strategy": "test"}
    path = tmp_path / "sft.json"
    export_sft([item], path, "alpaca")
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved[0]["output"].startswith("A")


def test_chunking_and_chinese_rouge():
    chunks = chunk_text("甲" * 250 + "\n\n" + "乙" * 250, "sample.txt", 300, 50)
    assert len(chunks) >= 2
    assert all(len(c.text) <= 300 for c in chunks)
    assert rouge_l("甲平台搭载甲武器", "甲平台搭载甲武器") == 1.0


def test_extractor_training_and_metrics():
    item = {"id": "x", "text": "甲平台搭载甲武器。",
            "entities": [{"name": "甲平台", "type": "Platform/Carrier"},
                         {"name": "甲武器", "type": "Weapon System"}],
            "relations": [{"source": "甲平台", "target": "甲武器", "type": "Equip-Carry",
                           "evidence": "甲平台搭载甲武器"}]}
    records = prepare_extractor_sft([item])
    assert "Equip-Carry" in records[0]["output"]
    metrics = evaluate_extractions([item], [item])
    assert metrics["entity"]["f1"] == 1.0
    assert metrics["relation"]["f1"] == 1.0


def test_generation_pipeline_with_injected_model():
    kg = fixture_graph()
    edge = next(e for e in kg.edges("Equip-Carry") if kg.graph.nodes[e[1]]["name"] == "甲武器")
    subgraph = Subgraph("test", (edge[0], edge[1]), (edge[2],), 1)

    class FakeModel:
        def complete(self, system, user, temperature):
            assert temperature == 0.7
            return {"question": "甲平台所搭载的武器是哪一项？", "options": [
                {"label": "A", "text": "甲 武器"}, {"label": "B", "text": "乙武器"},
                {"label": "C", "text": "丙武器"}, {"label": "D", "text": "丁武器"}],
                "answer": ["A"], "explanation": "甲平台搭载甲武器。",
                "supporting_facts": [edge[2]], "referenced_entities": [edge[0], edge[1]],
                "answer_entities": [edge[1]]}

    items, stats = generate_qa(kg, [subgraph], FakeModel(), ("single_choice",))
    assert stats["accepted"] == 1
    assert items[0]["type"] == "single_choice"


def test_source_grounded_spec_materialization():
    kg = MilitaryGraph()
    kg.add_entity({"name": "甲武器", "type": "Weapon System", "confidence": 0.9,
                   "attributes": {"caliber": "10毫米", "weight": "50千克"}, "chunk_id": "c1"})
    added = materialize_specifications(kg, [Chunk("c1", "x.txt", "甲武器的口径为10毫米。")])
    assert added == 1
    edge = next(kg.edges("Weapon-Spec"))
    assert edge[3]["evidence"][0]["text"] in "甲武器的口径为10毫米。"


def test_generation_resume_skips_completed_attempts():
    kg = fixture_graph()
    edge = next(kg.edges("Equip-Carry"))
    subgraph = Subgraph("test", (edge[0], edge[1]), (edge[2],), 1)

    class NeverCalled:
        def complete(self, *args):
            raise AssertionError("completed attempt should be skipped")

    items, stats = generate_qa(kg, [subgraph], NeverCalled(), ("single_choice",),
                               existing_stats={"attempted": 1, "accepted": 0,
                                               "invalid": 1, "duplicates": 0})
    assert items == [] and stats["attempted"] == 1


def test_offline_evaluation():
    kg = fixture_graph()
    selected = traverse(kg)
    metrics = traversal_metrics(kg, selected)
    assert metrics["structural_types"] >= 1
    baseline = random_edge_baseline(kg, selected, seed=1)
    assert len(baseline) <= len(selected)
    qa = evaluate_qa([{"id": "1", "type": "single_choice", "answer": ["A"]}],
                     [{"id": "1", "answer": ["A"]}])
    assert qa["accuracy"] == 1.0
