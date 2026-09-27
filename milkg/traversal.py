"""Five military semantic traversal rules from MilKG-QA Section III-E."""

from dataclasses import dataclass, asdict
from collections import defaultdict, deque

from .graph import MilitaryGraph


@dataclass(frozen=True)
class Subgraph:
    strategy: str
    nodes: tuple[str, ...]
    edges: tuple[str, ...]
    depth: int
    score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _candidate(strategy: str, edges: list[tuple[str, str, str, dict]]) -> Subgraph:
    nodes = tuple(dict.fromkeys(n for source, target, _, _ in edges for n in (source, target)))
    return Subgraph(strategy, nodes, tuple(e[2] for e in edges), len(edges))


def _index(kg: MilitaryGraph):
    index = defaultdict(list)
    for edge in kg.edges():
        index[edge[3]["type"]].append(edge)
    return index


def equipment_chains(kg: MilitaryGraph) -> list[Subgraph]:
    ix = _index(kg)
    result = []
    for carry in ix["Equip-Carry"]:
        platform, weapon = carry[:2]
        for unit_equip in ix["Unit-Equip"]:
            unit, equipped = unit_equip[:2]
            if equipped not in {platform, weapon}:
                continue
            base = [carry, unit_equip]
            result.append(_candidate("equipment_chain", base))
            for org in ix["Unit-Org"]:
                if org[0] == unit:
                    result.append(_candidate("equipment_chain", base + [org]))
    return result


def conflict_chains(kg: MilitaryGraph) -> list[Subgraph]:
    ix = _index(kg)
    result = []
    for counter in ix["Equip-Counter"]:
        weapon_a, weapon_b = counter[:2]
        for carry in ix["Equip-Carry"]:
            if carry[1] != weapon_b:
                continue
            base = [counter, carry]
            result.append(_candidate("conflict_chain", base))
            for spec in ix["Weapon-Spec"]:
                if spec[0] == weapon_a:
                    result.append(_candidate("conflict_chain", base + [spec]))
    return result


def campaign_panoramas(kg: MilitaryGraph) -> list[Subgraph]:
    ix = _index(kg)
    result = []
    for campaign, data in kg.graph.nodes(data=True):
        if data["type"] != "Campaign/Operation":
            continue
        participants = [e for e in ix["Campaign-Part"] if e[0] == campaign]
        consequences = [e for e in ix["Causal-Lead"] if e[0] == campaign]
        for part in participants:
            unit = part[1]
            equipment = [e for e in ix["Unit-Equip"] if e[0] == unit]
            for equip in equipment or [None]:
                base = [part] + ([equip] if equip else [])
                for causal in consequences or [None]:
                    path = base + ([causal] if causal else [])
                    if len(path) >= 2:
                        result.append(_candidate("campaign_panorama", path))
    return result


def entity_comparisons(kg: MilitaryGraph) -> list[Subgraph]:
    ix = _index(kg)
    by_type = defaultdict(list)
    for edge in ix["Weapon-Spec"]:
        by_type[kg.graph.nodes[edge[0]]["type"]].append(edge)
    result = []
    for edges in by_type.values():
        for i, left in enumerate(edges):
            for right in edges[i + 1:]:
                if left[0] != right[0]:
                    result.append(_candidate("entity_comparison", [left, right]))
    return result


def atomic_facts(kg: MilitaryGraph) -> list[Subgraph]:
    """One-hop slices for the paper's easy QA difficulty tier."""
    return [Subgraph("atomic_fact", (source, target), (edge_id,), 1,
                     round(float(data["confidence"]), 6))
            for source, target, edge_id, data in kg.edges()]


def multi_hop_paths(kg: MilitaryGraph, min_depth: int = 2, max_depth: int = 4,
                    max_paths: int = 2000) -> list[Subgraph]:
    if min_depth < 2 or max_depth < min_depth:
        raise ValueError("Require 2 <= min_depth <= max_depth")
    adjacency = defaultdict(list)
    for edge in kg.edges():
        adjacency[edge[0]].append((edge[1], edge))
        adjacency[edge[1]].append((edge[0], edge))
    result: list[Subgraph] = []
    seen: set[tuple[str, ...]] = set()
    for seed in sorted(kg.graph):
        queue = deque([(seed, [seed], [])])
        while queue and len(result) < max_paths:
            current, nodes, edges = queue.popleft()
            if min_depth <= len(edges) <= max_depth:
                key = tuple(sorted(e[2] for e in edges))
                if key not in seen:
                    seen.add(key)
                    result.append(_candidate("multi_hop", edges))
            if len(edges) == max_depth:
                continue
            for neighbor, edge in adjacency[current]:
                if neighbor not in nodes:
                    queue.append((neighbor, nodes + [neighbor], edges + [edge]))
        if len(result) >= max_paths:
            break
    return result


def score_subgraph(kg: MilitaryGraph, subgraph: Subgraph) -> float:
    types = {kg.graph.nodes[n]["type"] for n in subgraph.nodes}
    density = len(types) / max(1, len(subgraph.nodes))
    depth = 1.0 if subgraph.depth in {2, 3} else (0.7 if subgraph.depth == 4 else 0.4)
    confidence = sum(kg.edge(e)[2]["confidence"] for e in subgraph.edges) / len(subgraph.edges)
    return round(0.4 * density + 0.25 * depth + 0.35 * confidence, 6)


def traverse(kg: MilitaryGraph, strategies: tuple[str, ...] = (
    "equipment_chain", "conflict_chain", "campaign_panorama", "entity_comparison", "multi_hop"
), min_depth: int = 2, max_depth: int = 4, max_paths: int = 2000,
             max_subgraphs: int = 500, max_overlap: float = 0.5) -> list[Subgraph]:
    available = {
        "equipment_chain": equipment_chains,
        "conflict_chain": conflict_chains,
        "campaign_panorama": campaign_panoramas,
        "entity_comparison": entity_comparisons,
        "multi_hop": lambda graph: multi_hop_paths(graph, min_depth, max_depth, max_paths),
    }
    unknown = set(strategies) - set(available)
    if unknown:
        raise ValueError(f"Unknown strategies: {sorted(unknown)}")
    if not 0 <= max_overlap <= 1:
        raise ValueError("max_overlap must be between 0 and 1")
    if max_subgraphs < 1:
        return []
    candidates = []
    seen = set()
    for strategy in strategies:
        for subgraph in available[strategy](kg):
            key = (strategy, tuple(sorted(subgraph.edges)))
            if key in seen:
                continue
            seen.add(key)
            candidates.append(Subgraph(subgraph.strategy, subgraph.nodes, subgraph.edges,
                                       subgraph.depth, score_subgraph(kg, subgraph)))
    candidates.sort(key=lambda s: (-s.score, s.strategy, s.edges))
    selected: list[Subgraph] = []
    # Preserve at least one path from each applicable cognitive rule. Otherwise
    # a generic BFS path with the same nodes can erase the semantic rule's
    # output before QA generation, even though the relation pattern differs.
    for strategy in strategies:
        first = next((candidate for candidate in candidates if candidate.strategy == strategy), None)
        if first is not None and len(selected) < max_subgraphs:
            selected.append(first)
    for subgraph in candidates:
        if len(selected) >= max_subgraphs:
            break
        if subgraph in selected:
            continue
        node_set = set(subgraph.nodes)
        if any(len(node_set & set(prior.nodes)) / min(len(node_set), len(prior.nodes)) >= max_overlap
               for prior in selected):
            continue
        selected.append(subgraph)
        if len(selected) >= max_subgraphs:
            break
    return selected
