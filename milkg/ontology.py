"""The 14 entity types and 12 directed relations in MilKG-QA Table I.

Some endpoint constraints are not specified by the paper. The choices below
are explicit engineering interpretations, documented in README.md.
"""

from dataclasses import dataclass

ENTITY_TYPES = (
    "Weapon System", "Platform/Carrier", "Ammunition", "Electronic Equipment",
    "Combat Unit", "Command Structure", "Logistics Support",
    "Military Personnel", "Tactic/Method", "Campaign/Operation",
    "Military Facility", "Geographic Location", "Temporal Node",
    "Technical Specification",
)


@dataclass(frozen=True)
class RelationSpec:
    sources: frozenset[str]
    targets: frozenset[str]
    description: str


def spec(sources: str, targets: str, description: str) -> RelationSpec:
    return RelationSpec(frozenset(sources.split("|")), frozenset(targets.split("|")), description)


RELATIONS = {
    "Equip-Carry": spec("Platform/Carrier", "Weapon System|Ammunition|Electronic Equipment", "platform carries equipment"),
    "Equip-Counter": spec("Weapon System|Ammunition|Electronic Equipment", "Weapon System|Ammunition|Electronic Equipment", "equipment counters equipment"),
    "Equip-Develop": spec("Weapon System|Ammunition|Electronic Equipment|Platform/Carrier", "Combat Unit|Command Structure|Military Facility", "equipment developed by organization/facility"),
    "Unit-Org": spec("Combat Unit", "Combat Unit|Command Structure", "unit belongs to higher organization"),
    "Unit-Equip": spec("Combat Unit", "Weapon System|Ammunition|Electronic Equipment|Platform/Carrier", "unit operates equipment"),
    "Person-Cmd": spec("Military Personnel", "Combat Unit|Command Structure|Campaign/Operation", "person commands unit or operation"),
    "Tactic-Apply": spec("Tactic/Method", "Campaign/Operation|Combat Unit|Weapon System|Platform/Carrier", "tactic applied to operation/unit/equipment"),
    "Tactic-Counter": spec("Tactic/Method", "Tactic/Method|Weapon System|Platform/Carrier|Combat Unit", "tactic counters another tactic or force"),
    "Campaign-Part": spec("Campaign/Operation", "Combat Unit|Command Structure|Military Personnel|Weapon System|Platform/Carrier", "campaign has participant"),
    "Weapon-Spec": spec("Weapon System|Ammunition|Electronic Equipment|Platform/Carrier", "Technical Specification", "equipment has technical specification"),
    "Facility-Loc": spec("Military Facility", "Geographic Location", "facility is at location"),
    "Causal-Lead": spec("Campaign/Operation|Tactic/Method|Temporal Node", "Campaign/Operation|Tactic/Method|Temporal Node", "event or tactic leads to consequence"),
}


def valid_relation(kind: str, source_type: str, target_type: str) -> bool:
    rule = RELATIONS.get(kind)
    return bool(rule and source_type in rule.sources and target_type in rule.targets)
