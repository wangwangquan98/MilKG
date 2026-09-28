import json

from milkg.documents import Chunk, chunk_text, preprocess_text
from milkg.extraction import RELATION_GUIDANCE, extract_chunk, validate_extraction
from milkg.graph import MilitaryGraph
from milkg.ontology import ENTITY_TYPES, RELATIONS
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
    assert len(ENTITY_TYPES) == 14 and len(RELATIONS) == 12
    assert set(RELATION_GUIDANCE) == set(RELATIONS)
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


def test_ocr_preprocessing_before_chunking_preserves_paragraphs():
    raw = "前苏 联研 制了MG 34式机枪。\n射 程为300 米。\n\n第二 段保留。"
    cleaned = preprocess_text(raw)
    assert cleaned == "前苏联研制了MG34式机枪。射程为300米。\n\n第二段保留。"
    chunks = chunk_text(raw, "ocr.txt", 300, 20)
    assert len(chunks) == 1
    assert chunks[0].text == cleaned


def test_long_ocr_paragraph_splits_at_sentence_boundaries():
    text = "甲平台搭载甲武器。" * 60
    chunks = chunk_text(text, "long.txt", 120, 20)
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 120 and chunk.text.endswith("。") for chunk in chunks)


def test_relation_review_recovers_ocr_spaced_evidence():
    chunk = Chunk("x:1", "ocr.txt", "甲平台搭载甲 武器。")

    class TwoPassModel:
        calls = 0

        def complete(self, system, user, temperature):
            self.calls += 1
            if self.calls == 1:
                return {"entities": [{"name": "甲平台", "type": "Platform/Carrier"},
                                     {"name": "甲武器", "type": "Weapon System"}],
                        "relations": []}
            assert "甲武器 (Weapon System)" in user
            return {"relations": [{"source": "甲平台", "target": "甲武器",
                                   "type": "Equip-Carry", "evidence": "甲平台搭载甲武器"}]}

    model = TwoPassModel()
    phases = []
    result = extract_chunk(chunk, model, on_phase=phases.append)
    assert model.calls == 2
    assert "复查关系" in phases[0]
    assert result["relations"][0]["evidence"] == "甲平台搭载甲 武器"
    assert result["diagnostics"]["focused_relation_candidates"] == 1


def test_country_of_origin_is_not_development_relation():
    chunk = Chunk("x:2", "source.txt", "某兵工厂的甲武器很有名。")
    raw = {"entities": [{"name": "某兵工厂", "type": "Military Facility"},
                        {"name": "甲武器", "type": "Weapon System"}],
           "relations": [{"source": "甲武器", "target": "某兵工厂",
                          "type": "Equip-Develop", "evidence": "某兵工厂的甲武器"}]}
    result = validate_extraction(raw, chunk)
    assert result["relations"] == []
    assert result["diagnostics"]["relation_rejections"]["no_development_statement"] == 1


def test_generic_nationality_or_dynasty_cannot_be_developer():
    for target in ("德国人", "元朝"):
        text = f"{target}研制了甲武器。"
        chunk = Chunk("x:3", "source.txt", text)
        raw = {"entities": [{"name": target, "type": "Command Structure"},
                            {"name": "甲武器", "type": "Weapon System"}],
               "relations": [{"source": "甲武器", "target": target,
                              "type": "Equip-Develop", "evidence": text}]}
        result = validate_extraction(raw, chunk)
        assert result["relations"] == []
        assert result["diagnostics"]["relation_rejections"]["generic_developer"] == 1

    valid_chunk = Chunk("x:4", "source.txt", "甲兵工厂设计了甲武器。")
    valid_raw = {"entities": [{"name": "甲兵工厂", "type": "Military Facility"},
                              {"name": "甲武器", "type": "Weapon System"}],
                 "relations": [{"source": "甲武器", "target": "甲兵工厂",
                                "type": "Equip-Develop", "evidence": valid_chunk.text}]}
    assert len(validate_extraction(valid_raw, valid_chunk)["relations"]) == 1


def test_superior_or_copied_equipment_is_not_a_counter_relation():
    entities = [{"name": "甲武器", "type": "Weapon System"},
                {"name": "乙武器", "type": "Weapon System"}]
    for statement in ("甲武器性能优于乙武器。", "甲武器仿制了乙武器。"):
        chunk = Chunk("x:5", "source.txt", statement)
        raw = {"entities": entities,
               "relations": [{"source": "甲武器", "target": "乙武器",
                              "type": "Equip-Counter", "evidence": statement}]}
        result = validate_extraction(raw, chunk)
        assert result["relations"] == []
        assert result["diagnostics"]["relation_rejections"]["no_counter_statement"] == 1

    chunk = Chunk("x:6", "source.txt", "甲武器能够击毁乙武器。")
    raw["relations"][0]["evidence"] = chunk.text
    assert len(validate_extraction(raw, chunk)["relations"]) == 1


def test_generic_equipment_statement_does_not_prove_specific_unit_equipment():
    chunk = Chunk("x:7", "source.txt", "巴祖卡是火箭筒。火箭筒成为步兵班的骨干火力。")
    raw = {"entities": [{"name": "步兵班", "type": "Combat Unit"},
                        {"name": "巴祖卡", "type": "Weapon System"}],
           "relations": [{"source": "步兵班", "target": "巴祖卡", "type": "Unit-Equip",
                          "evidence": "火箭筒成为步兵班的骨干火力"}]}
    result = validate_extraction(raw, chunk)
    assert result["relations"] == []
    assert result["diagnostics"]["relation_rejections"]["endpoint_not_in_evidence"] == 1

    explicit = Chunk("x:8", "source.txt", "步兵班装备巴祖卡。")
    raw["relations"][0]["evidence"] = explicit.text
    assert len(validate_extraction(raw, explicit)["relations"]) == 1


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
    distractor_claim = {**raw, "explanation": "甲平台搭载甲武器，乙武器不是其装备。"}
    assert validate_qa(distractor_claim, kg, subgraph, "single_choice")[1] == "explanation discusses distractor"
    item = {**raw, "type": "single_choice", "style": "formal_exam",
            "difficulty": "easy", "strategy": "test"}
    path = tmp_path / "sft.json"
    export_sft([item], path, "alpaca")
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved[0]["output"].startswith("A")
    assert set(saved[0]) == {"instruction", "input", "output"}
    export_sft([item], path, "sharegpt")
    assert set(json.loads(path.read_text(encoding="utf-8"))[0]) == {"conversations"}
    export_sft([item], path, "chatml")
    assert set(json.loads(path.read_text(encoding="utf-8"))[0]) == {"messages"}


def test_training_text_rejects_internal_ids_and_external_context(tmp_path):
    kg = fixture_graph()
    edge = next(kg.edges("Equip-Carry"))
    subgraph = Subgraph("test", (edge[0], edge[1]), (edge[2],), 1)
    base = {"question": "甲平台所搭载的武器是哪一项？", "options": [
        {"label": "A", "text": "甲武器"}, {"label": "B", "text": "乙武器"},
        {"label": "C", "text": "丙武器"}, {"label": "D", "text": "丁武器"}],
        "answer": ["A"], "explanation": "甲平台搭载甲武器。",
        "supporting_facts": [edge[2]], "referenced_entities": [edge[0], edge[1]],
        "answer_entities": [edge[1]]}
    for changed in ({"explanation": f"根据子图事实{edge[2]}，甲平台搭载甲武器。"},
                    {"question": "根据给定资料，甲平台搭载哪种武器？"},
                    {"options": [{**base["options"][0], "text": "甲武器（n000001）"},
                                 *base["options"][1:]]}):
        item = {**base, **changed}
        assert validate_qa(item, kg, subgraph, "single_choice")[1] == "graph reference in training text"
        path = tmp_path / "sft.json"
        try:
            export_sft([item], path)
            assert False, "leaking record should not be exported"
        except ValueError:
            assert not path.exists()


def test_generation_rewrites_graph_references_before_accepting():
    kg = fixture_graph()
    edge = next(kg.edges("Equip-Carry"))
    subgraph = Subgraph("test", (edge[0], edge[1]), (edge[2],), 1)

    class RewritingModel:
        calls = 0

        def complete(self, system, user, temperature):
            self.calls += 1
            if self.calls == 1:
                return {"question": "甲平台所搭载的武器是哪一项？", "options": [
                    {"label": "A", "text": "甲武器"}, {"label": "B", "text": "乙武器"},
                    {"label": "C", "text": "丙武器"}, {"label": "D", "text": "丁武器"}],
                    "answer": ["A"], "explanation": f"根据子图事实{edge[2]}，甲平台搭载甲武器。",
                    "supporting_facts": [edge[2]], "referenced_entities": [edge[0], edge[1]],
                    "answer_entities": [edge[1]]}
            assert "独立问答" in system
            return {"question": "甲平台所搭载的武器是哪一项？", "explanation": "甲平台搭载甲武器。"}

    model = RewritingModel()
    items, stats = generate_qa(kg, [subgraph], model, ("single_choice",))
    assert model.calls == 2
    assert stats["accepted"] == stats["rewritten"] == 1
    assert items[0]["explanation"] == "甲平台搭载甲武器。"


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
