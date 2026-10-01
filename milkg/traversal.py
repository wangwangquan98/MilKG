"""Five military semantic traversal rules from MilKG-QA Section III-E."""

from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from collections.abc import Callable, Iterator
from itertools import combinations

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


Edge = tuple[str, str, str, dict]


class _TraversalIndex:
    """Build once per traversal; indexes retain the graph's edge order."""

    def __init__(self, kg: MilitaryGraph):
        self.edges = list(kg.edges())
        self.by_type: dict[str, list[Edge]] = defaultdict(list)
        self.by_source: dict[tuple[str, str], list[Edge]] = defaultdict(list)
        self.by_target: dict[tuple[str, str], list[Edge]] = defaultdict(list)
        self.node_types = {node: data["type"] for node, data in kg.graph.nodes(data=True)}
        self.confidence = {}
        self.order = {}
        for position, edge in enumerate(self.edges):
            source, target, edge_id, data = edge
            kind = data["type"]
            self.by_type[kind].append(edge)
            self.by_source[kind, source].append(edge)
            self.by_target[kind, target].append(edge)
            self.confidence[edge_id] = data["confidence"]
            self.order[edge_id] = position


def equipment_chains(kg: MilitaryGraph) -> list[Subgraph]:
    return list(_equipment_chains(_TraversalIndex(kg)))


def _equipment_chains(ix: _TraversalIndex) -> Iterator[Subgraph]:
    for carry in ix.by_type["Equip-Carry"]:
        platform, weapon = carry[:2]
        equipped = list(ix.by_target.get(("Unit-Equip", platform), ()))
        if weapon != platform:
            equipped.extend(ix.by_target.get(("Unit-Equip", weapon), ()))
        equipped.sort(key=lambda edge: ix.order[edge[2]])
        for unit_equip in equipped:
            unit = unit_equip[0]
            base = [carry, unit_equip]
            yield _candidate("equipment_chain", base)
            for org in ix.by_source.get(("Unit-Org", unit), ()):
                yield _candidate("equipment_chain", base + [org])


def conflict_chains(kg: MilitaryGraph) -> list[Subgraph]:
    return list(_conflict_chains(_TraversalIndex(kg)))


def _conflict_chains(ix: _TraversalIndex) -> Iterator[Subgraph]:
    for counter in ix.by_type["Equip-Counter"]:
        weapon_a, weapon_b = counter[:2]
        for carry in ix.by_target.get(("Equip-Carry", weapon_b), ()):
            base = [counter, carry]
            yield _candidate("conflict_chain", base)
            for spec in ix.by_source.get(("Weapon-Spec", weapon_a), ()):
                yield _candidate("conflict_chain", base + [spec])


def campaign_panoramas(kg: MilitaryGraph) -> list[Subgraph]:
    return list(_campaign_panoramas(_TraversalIndex(kg)))


def _campaign_panoramas(ix: _TraversalIndex) -> Iterator[Subgraph]:
    for campaign, kind in ix.node_types.items():
        if kind != "Campaign/Operation":
            continue
        participants = ix.by_source.get(("Campaign-Part", campaign), ())
        consequences = ix.by_source.get(("Causal-Lead", campaign), ())
        for part in participants:
            unit = part[1]
            equipment = ix.by_source.get(("Unit-Equip", unit), ())
            for equip in equipment or [None]:
                base = [part] + ([equip] if equip else [])
                for causal in consequences or [None]:
                    path = base + ([causal] if causal else [])
                    if len(path) >= 2:
                        yield _candidate("campaign_panorama", path)


def entity_comparisons(kg: MilitaryGraph) -> list[Subgraph]:
    return list(_entity_comparisons(_TraversalIndex(kg)))


def _entity_comparisons(ix: _TraversalIndex) -> Iterator[Subgraph]:
    by_type = defaultdict(list)
    for edge in ix.by_type["Weapon-Spec"]:
        by_type[ix.node_types[edge[0]]].append(edge)
    for edges in by_type.values():
        for left, right in combinations(edges, 2):
            if left[0] != right[0]:
                yield _candidate("entity_comparison", [left, right])


def atomic_facts(kg: MilitaryGraph) -> list[Subgraph]:
    """One-hop slices for the paper's easy QA difficulty tier."""
    return [Subgraph("atomic_fact", (source, target), (edge_id,), 1,
                     round(float(data["confidence"]), 6))
            for source, target, edge_id, data in kg.edges()]


def multi_hop_paths(kg: MilitaryGraph, min_depth: int = 2, max_depth: int = 4,
                    max_paths: int = 2000) -> list[Subgraph]:
    return list(_multi_hop_paths(_TraversalIndex(kg), min_depth, max_depth, max_paths))


def _multi_hop_paths(ix: _TraversalIndex, min_depth: int, max_depth: int,
                     max_paths: int) -> Iterator[Subgraph]:
    if min_depth < 2 or max_depth < min_depth:
        raise ValueError("Require 2 <= min_depth <= max_depth")
    adjacency = defaultdict(list)
    for edge in ix.edges:
        adjacency[edge[0]].append((edge[1], edge))
        adjacency[edge[1]].append((edge[0], edge))
    count = 0
    seen: set[tuple[str, ...]] = set()
    # Isolated nodes cannot yield a path.
    for seed in sorted(adjacency):
        queue = deque([(seed, [seed], [])])
        while queue and count < max_paths:
            current, nodes, edges = queue.popleft()
            if min_depth <= len(edges) <= max_depth:
                key = tuple(sorted(e[2] for e in edges))
                if key not in seen:
                    seen.add(key)
                    count += 1
                    yield _candidate("multi_hop", edges)
                    if count >= max_paths:
                        return
            if len(edges) == max_depth:
                continue
            for neighbor, edge in adjacency[current]:
                if neighbor not in nodes:
                    queue.append((neighbor, nodes + [neighbor], edges + [edge]))
        if count >= max_paths:
            break


def score_subgraph(kg: MilitaryGraph, subgraph: Subgraph) -> float:
    ix = _TraversalIndex(kg)
    return _score_subgraph(ix, subgraph)


def _score_subgraph(ix: _TraversalIndex, subgraph: Subgraph) -> float:
    types = {ix.node_types[n] for n in subgraph.nodes}
    density = len(types) / max(1, len(subgraph.nodes))
    depth = 1.0 if subgraph.depth in {2, 3} else (0.7 if subgraph.depth == 4 else 0.4)
    confidence = sum(ix.confidence[e] for e in subgraph.edges) / len(subgraph.edges)
    return round(0.4 * density + 0.25 * depth + 0.35 * confidence, 6)


def traverse(kg: MilitaryGraph, strategies: tuple[str, ...] = (
    "equipment_chain", "conflict_chain", "campaign_panorama", "entity_comparison", "multi_hop"
), min_depth: int = 2, max_depth: int = 4, max_paths: int = 2000,
             max_subgraphs: int = 500, max_overlap: float = 0.5,
             on_progress: Callable[[str], None] | None = None) -> list[Subgraph]:
    available = {
        "equipment_chain": _equipment_chains,
        "conflict_chain": _conflict_chains,
        "campaign_panorama": _campaign_panoramas,
        "entity_comparison": _entity_comparisons,
        "multi_hop": lambda ix: _multi_hop_paths(ix, min_depth, max_depth, max_paths),
    }
    unknown = set(strategies) - set(available)
    if unknown:
        raise ValueError(f"Unknown strategies: {sorted(unknown)}")
    if not 0 <= max_overlap <= 1:
        raise ValueError("max_overlap must be between 0 and 1")
    if max_subgraphs < 1:
        return []
    ix = _TraversalIndex(kg)
    labels = {"equipment_chain": "装备链", "conflict_chain": "对抗链",
              "campaign_panorama": "战役全景", "entity_comparison": "实体比较",
              "multi_hop": "多跳路径"}
    candidates = []
    seen = set()
    for strategy in strategies:
        if on_progress:
            on_progress(f"正在遍历{labels[strategy]}，累计 {len(candidates)} 个候选子图。")
        before = len(candidates)
        for subgraph in available[strategy](ix):
            key = (strategy, tuple(sorted(subgraph.edges)))
            if key in seen:
                continue
            seen.add(key)
            candidates.append(Subgraph(subgraph.strategy, subgraph.nodes, subgraph.edges,
                                       subgraph.depth, _score_subgraph(ix, subgraph)))
            if on_progress and len(candidates) % 10000 == 0:
                on_progress(f"正在遍历{labels[strategy]}，已评分 {len(candidates)} 个候选子图。")
        if on_progress:
            on_progress(f"{labels[strategy]}遍历完成：新增 {len(candidates) - before} 个候选子图。")
    if on_progress:
        on_progress(f"候选评分完成，正在排序和去重 {len(candidates)} 个子图。")
    candidates.sort(key=lambda s: (-s.score, s.strategy, s.edges))
    selected: list[Subgraph] = []
    selected_set = set()
    node_postings: dict[str, list[int]] = defaultdict(list)
    selected_sizes: list[int] = []

    def select(subgraph: Subgraph) -> None:
        position = len(selected)
        selected.append(subgraph)
        selected_set.add(subgraph)
        selected_sizes.append(len(subgraph.nodes))
        for node in subgraph.nodes:
            node_postings[node].append(position)

    def overlaps(subgraph: Subgraph) -> bool:
        if max_overlap == 0:
            return bool(selected)
        shared: dict[int, int] = {}
        for node in subgraph.nodes:
            for position in node_postings.get(node, ()):
                count = shared.get(position, 0) + 1
                if count / min(len(subgraph.nodes), selected_sizes[position]) >= max_overlap:
                    return True
                shared[position] = count
        return False
    # Preserve at least one path from each applicable cognitive rule. Otherwise
    # a generic BFS path with the same nodes can erase the semantic rule's
    # output before QA generation, even though the relation pattern differs.
    first_by_strategy = {}
    for candidate in candidates:
        first_by_strategy.setdefault(candidate.strategy, candidate)
    for strategy in strategies:
        first = first_by_strategy.get(strategy)
        if first is not None and len(selected) < max_subgraphs:
            select(first)
    for position, subgraph in enumerate(candidates, 1):
        if on_progress and position % 10000 == 0:
            on_progress(f"已筛选 {position}/{len(candidates)} 个候选，保留 {len(selected)} 个语义子图。")
        if len(selected) >= max_subgraphs:
            break
        if subgraph in selected_set:
            continue
        if overlaps(subgraph):
            continue
        select(subgraph)
        if len(selected) >= max_subgraphs:
            break
    if on_progress:
        on_progress(f"子图筛选完成：{len(candidates)} 个候选中保留 {len(selected)} 个语义子图。")
    return selected
