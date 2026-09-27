"""Materialize source-grounded equipment attributes as Weapon-Spec KG edges."""

import re

from .documents import Chunk
from .graph import MilitaryGraph


EQUIPMENT_TYPES = {"Weapon System", "Platform/Carrier", "Ammunition", "Electronic Equipment"}
SPEC_KEYS = {
    "caliber", "weight", "projectile_weight", "projectile", "mechanism", "ammunition_type",
    "effective_range", "range", "rate_of_fire", "capacity", "firing_rate", "velocity",
    "口径", "重量", "射程", "有效射程", "对空有效射程", "对地有效射程",
    "自动方式", "射击模式", "性能", "威力", "结构", "装填方式", "弹药类型",
}


def _compact_with_positions(text: str) -> tuple[str, list[int]]:
    chars, positions = [], []
    for i, char in enumerate(text):
        if not char.isspace():
            chars.append(char)
            positions.append(i)
    return "".join(chars), positions


def _evidence_span(text: str, name: str, value: str, max_gap: int = 160) -> str | None:
    compact, positions = _compact_with_positions(text)
    needle = re.sub(r"\s+", "", value)
    entity = re.sub(r"\s+", "", name)
    if not needle or not entity:
        return None
    value_positions = [m.start() for m in re.finditer(re.escape(needle), compact)]
    if not value_positions:
        return None
    entity_positions = [m.start() for m in re.finditer(re.escape(entity), compact)]
    if not entity_positions:
        return None
    value_at, entity_at = min(((v, e) for v in value_positions for e in entity_positions),
                              key=lambda pair: abs(pair[0] - pair[1]))
    if abs(entity_at - value_at) > max_gap:
        return None
    start = positions[min(value_at, entity_at)]
    end = positions[max(value_at + len(needle), entity_at + len(entity)) - 1] + 1
    return text[start:end]


def materialize_specifications(kg: MilitaryGraph, chunks: list[Chunk]) -> int:
    """Only add an edge when entity and attribute value occur close in source text."""
    source = {chunk.id: chunk.text for chunk in chunks}
    additions = []
    for _, data in list(kg.graph.nodes(data=True)):
        if data["type"] not in EQUIPMENT_TYPES:
            continue
        for key, attribute in data.get("attributes", {}).items():
            if key.casefold() not in SPEC_KEYS:
                continue
            value = str(attribute["value"]).strip()
            chunk_id = attribute.get("chunk_id")
            text = source.get(chunk_id, "")
            evidence = _evidence_span(text, data["name"], value)
            if not evidence:
                continue
            additions.append((data["name"], data["type"], key, value,
                              float(attribute["confidence"]), chunk_id, evidence))
    before = kg.graph.number_of_edges()
    for name, kind, key, value, confidence, chunk_id, evidence in additions:
        spec_name = f"{key}：{value}"
        kg.add_extraction({
            "entities": [
                {"name": name, "type": kind, "confidence": confidence, "chunk_id": chunk_id},
                {"name": spec_name, "type": "Technical Specification", "confidence": confidence,
                 "attributes": {"parameter": key, "value": value}, "chunk_id": chunk_id},
            ],
            "relations": [{"source": name, "target": spec_name, "type": "Weapon-Spec",
                           "confidence": confidence, "evidence": evidence, "chunk_id": chunk_id,
                           "derived_from_attribute": True}],
        })
    return kg.graph.number_of_edges() - before
