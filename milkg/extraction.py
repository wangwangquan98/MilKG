"""Ontology-constrained entity/relation extraction with source evidence."""

from .documents import Chunk
from .llm import ChatModel
from .ontology import ENTITY_TYPES, RELATIONS, valid_relation


EXTRACTION_SYSTEM = """你是军事文献的实体关系抽取器。只提取文本明确陈述的事实，不补充常识。
输出单个 JSON 对象：{"entities": [{"name": str, "type": str, "aliases": [str],
"attributes": {str: str}, "confidence": 0..1}], "relations": [{"source": str,
"target": str, "type": str, "confidence": 0..1, "evidence": str}]}。
关系 evidence 必须是输入原文的连续片段；source 和 target 必须出现在 entities 中。
方向严格遵守关系定义，不确定就省略。不要输出 Markdown。"""


def extraction_prompt(chunk: Chunk) -> str:
    types = "、".join(ENTITY_TYPES)
    relations = "\n".join(
        f"{name}: {rule.description}; source={','.join(sorted(rule.sources))}; target={','.join(sorted(rule.targets))}"
        for name, rule in RELATIONS.items()
    )
    return f"实体类型：{types}\n关系类型及方向：\n{relations}\n文档片段 {chunk.id}：\n{chunk.text}"


def _confidence(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def validate_extraction(raw: dict, chunk: Chunk, min_confidence: float = 0.0) -> dict:
    """Discard ungrounded/ill-typed records, preserving provenance for accepted facts."""
    entities: list[dict] = []
    seen: dict[str, dict] = {}
    for item in raw.get("entities", []):
        if not isinstance(item, dict):
            continue
        name, kind = str(item.get("name", "")).strip(), item.get("type")
        confidence = _confidence(item.get("confidence", 1))
        if not name or kind not in ENTITY_TYPES or confidence < min_confidence or name not in chunk.text:
            continue
        if name in seen and seen[name]["confidence"] >= confidence:
            continue
        aliases = item.get("aliases", [])
        attributes = item.get("attributes", {})
        entity = {"name": name, "type": kind,
                  "aliases": [x for x in aliases if isinstance(x, str) and x.strip()] if isinstance(aliases, list) else [],
                  "attributes": {str(k): str(v) for k, v in attributes.items()} if isinstance(attributes, dict) else {},
                  "confidence": confidence, "chunk_id": chunk.id}
        seen[name] = entity
    entities = list(seen.values())
    relations: list[dict] = []
    for item in raw.get("relations", []):
        if not isinstance(item, dict):
            continue
        source, target, kind = item.get("source"), item.get("target"), item.get("type")
        evidence = str(item.get("evidence", "")).strip()
        confidence = _confidence(item.get("confidence", 1))
        if source not in seen or target not in seen or not evidence or evidence not in chunk.text:
            continue
        if confidence < min_confidence or not valid_relation(kind, seen[source]["type"], seen[target]["type"]):
            continue
        relations.append({"source": source, "target": target, "type": kind,
                          "confidence": confidence, "evidence": evidence, "chunk_id": chunk.id})
    return {"chunk_id": chunk.id, "entities": entities, "relations": relations}


def extract_chunk(chunk: Chunk, model: ChatModel, min_confidence: float = 0.0,
                  temperature: float = 0.1) -> dict:
    return validate_extraction(model.complete(EXTRACTION_SYSTEM, extraction_prompt(chunk), temperature),
                               chunk, min_confidence)
