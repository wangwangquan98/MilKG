"""Ontology-constrained entity/relation extraction with source evidence."""

import re
from collections import Counter, defaultdict
from collections.abc import Callable

from .documents import Chunk
from .llm import ChatModel
from .ontology import ENTITY_TYPES, RELATIONS, valid_relation


EXTRACTION_SYSTEM = """你是军事文献的实体关系抽取器。只提取文本明确陈述的事实，不补充常识。
输出单个 JSON 对象：{"entities": [{"name": str, "type": str, "aliases": [str],
"attributes": {str: str}, "confidence": 0..1}], "relations": [{"source": str,
"target": str, "type": str, "confidence": 0..1, "evidence": str}]}。
关系 evidence 必须是输入原文的连续片段（可忽略 OCR 空格）；source 和 target 必须出现在 entities 中。
国家和地区属于 Geographic Location，不能标为 Command Structure；司令部和指挥机关才属于 Command Structure。
具体人物、兵种人员和狙击手属于 Military Personnel；班、排、分队和军队组织才属于 Combat Unit。
兵工厂和基地属于 Military Facility，明确写出的口径、射程等数值可标为 Technical Specification。
逐一检查有原文依据的 12 类关系；能表达为关系的事实不要只放进 attributes。
发明者、发明年份、类别等若不符合任何关系定义，可保留为属性，不要捏造关系。
方向严格遵守关系定义，不确定就省略。不要输出 Markdown。"""

RELATION_GUIDANCE = {
    "Equip-Carry": "平台实际搭载或携带装备",
    "Equip-Counter": "一种装备在对抗中明确克制、打击或拦截另一装备；性能优于、仿制或借鉴不算克制",
    "Equip-Develop": "装备由明确写出的部队、指挥机构或军事设施研制；国家、朝代、国籍群体和个人发明者不能充当研发机构",
    "Unit-Org": "作战单位隶属于上级单位或指挥机构",
    "Unit-Equip": "作战单位使用或装备某武器、弹药、平台；证据必须同时点名单位和具体装备，泛指装备类别不能推断特定型号",
    "Person-Cmd": "军事人员指挥单位、机构或行动",
    "Tactic-Apply": "战术方法应用于行动、单位或装备",
    "Tactic-Counter": "战术方法明确克制另一方法、装备或单位",
    "Campaign-Part": "作战行动有明确的参与单位、人员或装备",
    "Weapon-Spec": "武器、弹药、电子装备或平台具有明确的技术规格",
    "Facility-Loc": "军事设施位于某地",
    "Causal-Lead": "行动、战术或时间节点明确导致另一行动、战术或时间节点",
}

RELATION_SYSTEM = """你是军事文献的实体关系抽取器，当前任务只复查关系。
只能在给定实体之间抽取原文明确陈述的关系，只使用列出的 12 种关系及其规定方向。
逐项检查关系类型，不要因为两个实体在同一段出现就建立关系；国别、发明者、年份和类别不应硬套现有关系。
例如“德国人德莱赛发明了某枪”不能写成“某枪由德国人研制”；“元朝制造出某枪”也不能把元朝标作研发机构。
Unit-Equip 的证据必须同时包含单位和具体装备名称；泛指火箭筒等类别的句子不能证明该单位使用某一具体型号。
evidence 必须来自给定文本，可以忽略 OCR 空格，但不可改写事实。返回单个 JSON 对象：
{"relations": [{"source": str, "target": str, "type": str, "confidence": 0..1, "evidence": str}]}。
没有符合定义的关系时返回 {"relations": []}。不要输出 Markdown。"""


def extraction_prompt(chunk: Chunk) -> str:
    types = "、".join(ENTITY_TYPES)
    relations = "\n".join(
        f"{name}: {RELATION_GUIDANCE[name]}；源类型={','.join(sorted(rule.sources))}；"
        f"目标类型={','.join(sorted(rule.targets))}"
        for name, rule in RELATIONS.items()
    )
    return f"实体类型：{types}\n关系类型及方向：\n{relations}\n文档片段 {chunk.id}：\n{chunk.text}"


def relation_prompt(chunk: Chunk, entities: list[dict]) -> str:
    relations = "\n".join(
        f"{name}: {RELATION_GUIDANCE[name]}；源类型={','.join(sorted(rule.sources))}；"
        f"目标类型={','.join(sorted(rule.targets))}"
        for name, rule in RELATIONS.items()
    )
    entity_list = "\n".join(f"- {entity['name']} ({entity['type']})" for entity in entities)
    return (f"允许的关系及方向：\n{relations}\n已核验实体：\n{entity_list}\n"
            f"文档片段 {chunk.id}：\n{chunk.text}")


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


def _source_span(text: str, fragment: str) -> str | None:
    """Return the literal source span when only OCR whitespace differs."""
    positions: list[int] = []
    chars: list[str] = []
    for index, char in enumerate(text):
        if not char.isspace():
            folded = char.casefold()
            chars.extend(folded)
            positions.extend([index] * len(folded))
    compact_text = "".join(chars)
    needle = _compact(fragment)
    if not needle:
        return None
    start = compact_text.find(needle)
    if start < 0:
        return None
    return text[positions[start]:positions[start + len(needle) - 1] + 1]


def _confidence(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _items(raw: dict, key: str) -> list:
    value = raw.get(key, [])
    return value if isinstance(value, list) else []


def _generic_developer(name: str) -> bool:
    """Reject country, nationality and dynasty names posing as developers."""
    return bool(re.fullmatch(
        r"(?:[\u3400-\u9fff]{1,8}国|[\u3400-\u9fff]{1,8}国人|"
        r"[\u3400-\u9fff]{1,8}朝|(?:前)?苏联)", name
    ))


def validate_extraction(raw: dict, chunk: Chunk, min_confidence: float = 0.0) -> dict:
    """Discard ungrounded/ill-typed records, preserving provenance for accepted facts."""
    entities: list[dict] = []
    seen: dict[str, dict] = {}
    entity_rejected: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    entity_candidates = _items(raw, "entities")
    relation_candidates = _items(raw, "relations")
    for item in entity_candidates:
        if not isinstance(item, dict):
            entity_rejected["invalid_record"] += 1
            continue
        name, kind = str(item.get("name", "")).strip(), item.get("type")
        confidence = _confidence(item.get("confidence", 1))
        source_name = _source_span(chunk.text, name)
        if not source_name:
            entity_rejected["name_not_in_source"] += 1
            continue
        if kind not in ENTITY_TYPES:
            entity_rejected["unknown_type"] += 1
            continue
        if confidence < min_confidence:
            entity_rejected["low_confidence"] += 1
            continue
        name = re.sub(r"\s+", "", source_name) if re.search(r"[\u3400-\u9fff]", source_name) else source_name
        if name in seen and seen[name]["confidence"] >= confidence:
            entity_rejected["duplicate"] += 1
            continue
        aliases = item.get("aliases", [])
        attributes = item.get("attributes", {})
        entity = {"name": name, "type": kind,
                  "aliases": [x for x in aliases if isinstance(x, str) and x.strip()] if isinstance(aliases, list) else [],
                  "attributes": {str(k): str(v) for k, v in attributes.items()} if isinstance(attributes, dict) else {},
                  "confidence": confidence, "chunk_id": chunk.id}
        seen[name] = entity
    entities = list(seen.values())
    names: dict[str, set[str]] = defaultdict(set)
    for entity in entities:
        for name in [entity["name"], *entity["aliases"]]:
            names[_compact(name)].add(entity["name"])

    def resolve(value: object) -> str | None:
        choices = names.get(_compact(value), set()) if isinstance(value, str) else set()
        return next(iter(choices)) if len(choices) == 1 else None

    relations: list[dict] = []
    unique: set[tuple[str, str, str, str]] = set()
    for item in relation_candidates:
        if not isinstance(item, dict):
            rejected["invalid_record"] += 1
            continue
        source, target = resolve(item.get("source")), resolve(item.get("target"))
        kind = item.get("type")
        evidence = _source_span(chunk.text, str(item.get("evidence", "")).strip())
        confidence = _confidence(item.get("confidence", 1))
        if source is None or target is None:
            rejected["unknown_endpoint"] += 1
            continue
        if not evidence:
            rejected["evidence_not_in_source"] += 1
            continue
        if confidence < min_confidence:
            rejected["low_confidence"] += 1
            continue
        if not valid_relation(kind, seen[source]["type"], seen[target]["type"]):
            rejected["invalid_type_or_direction"] += 1
            continue
        if kind == "Equip-Develop" and not re.search(
            r"研制|研发|开发|设计|制造|生产|发明|develop|design|manufactur|invent", evidence, re.I
        ):
            rejected["no_development_statement"] += 1
            continue
        if kind == "Equip-Develop" and _generic_developer(target):
            rejected["generic_developer"] += 1
            continue
        if kind == "Equip-Counter" and not re.search(
            r"克制|反制|对抗|击毁|摧毁|打击|击穿|拦截|压制|抵御|反坦克|"
            r"counter|defeat|destroy|intercept", evidence, re.I
        ):
            rejected["no_counter_statement"] += 1
            continue
        if kind == "Unit-Equip" and (
            not _source_span(evidence, source) or not _source_span(evidence, target)
        ):
            rejected["endpoint_not_in_evidence"] += 1
            continue
        key = (source, target, kind, evidence)
        if key in unique:
            continue
        unique.add(key)
        relations.append({"source": source, "target": target, "type": kind,
                          "confidence": confidence, "evidence": evidence, "chunk_id": chunk.id})
    return {"chunk_id": chunk.id, "entities": entities, "relations": relations,
            "diagnostics": {"entity_candidates": len(entity_candidates),
                            "relation_candidates": len(relation_candidates),
                            "entity_rejections": dict(entity_rejected),
                            "relation_rejections": dict(rejected)}}


def extract_chunk(chunk: Chunk, model: ChatModel, min_confidence: float = 0.0,
                  temperature: float = 0.1,
                  on_phase: Callable[[str], None] | None = None) -> dict:
    first_raw = model.complete(EXTRACTION_SYSTEM, extraction_prompt(chunk), temperature)
    first = validate_extraction(first_raw, chunk, min_confidence)
    first["diagnostics"]["initial_entity_candidates"] = len(_items(first_raw, "entities"))
    first["diagnostics"]["initial_relation_candidates"] = len(_items(first_raw, "relations"))
    first["diagnostics"]["focused_relation_candidates"] = 0
    if len(first["entities"]) < 2:
        return first
    if on_phase:
        on_phase(f"首轮识别 {len(first['entities'])} 个实体，正在复查关系")
    try:
        focused_raw = model.complete(RELATION_SYSTEM, relation_prompt(chunk, first["entities"]),
                                     temperature)
    except (RuntimeError, TimeoutError, OSError, ValueError) as exc:
        first["diagnostics"]["focused_pass_error"] = type(exc).__name__
        return first
    candidates = [*_items(first_raw, "relations"), *_items(focused_raw, "relations")]
    result = validate_extraction({"entities": first["entities"], "relations": candidates},
                                 chunk, min_confidence)
    result["diagnostics"]["initial_entity_candidates"] = first["diagnostics"]["initial_entity_candidates"]
    result["diagnostics"]["entity_rejections"] = first["diagnostics"]["entity_rejections"]
    result["diagnostics"]["initial_relation_candidates"] = first["diagnostics"]["initial_relation_candidates"]
    result["diagnostics"]["focused_relation_candidates"] = len(_items(focused_raw, "relations"))
    return result
