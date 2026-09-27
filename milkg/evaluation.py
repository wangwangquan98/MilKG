"""Offline structural diversity and downstream QA metrics."""

import random
from collections import defaultdict

from .graph import MilitaryGraph
from .ontology import ENTITY_TYPES
from .qa import rouge_l
from .traversal import Subgraph


def traversal_metrics(kg: MilitaryGraph, subgraphs: list[Subgraph]) -> dict:
    signatures = set()
    covered = set()
    depths = []
    for subgraph in subgraphs:
        signatures.add(tuple(kg.edge(edge)[2]["type"] for edge in subgraph.edges))
        covered.update(kg.graph.nodes[node]["type"] for node in subgraph.nodes)
        depths.append(subgraph.depth)
    mean = sum(depths) / len(depths) if depths else 0.0
    variance = sum((depth - mean) ** 2 for depth in depths) / len(depths) if depths else 0.0
    return {"subgraphs": len(subgraphs), "structural_types": len(signatures),
            "entity_type_coverage": len(covered) / len(ENTITY_TYPES),
            "average_depth": mean, "depth_variance": variance}


def random_edge_baseline(kg: MilitaryGraph, reference: list[Subgraph], seed: int = 42) -> list[Subgraph]:
    """Same starting nodes and depths, random edges (paper's Table III wording)."""
    rng = random.Random(seed)
    adjacency = defaultdict(list)
    for source, target, edge_id, _ in kg.edges():
        adjacency[source].append((target, edge_id))
        adjacency[target].append((source, edge_id))
    result = []
    for original in reference:
        if not original.nodes:
            continue
        nodes = [original.nodes[0]]
        edges = []
        while len(edges) < original.depth:
            choices = [(neighbor, edge_id) for neighbor, edge_id in adjacency[nodes[-1]]
                       if neighbor not in nodes]
            if not choices:
                break
            neighbor, edge_id = rng.choice(choices)
            nodes.append(neighbor)
            edges.append(edge_id)
        if len(edges) == original.depth:
            result.append(Subgraph("random_edge_baseline", tuple(nodes), tuple(edges), len(edges)))
    return result


def _canonical_answer(answer) -> str:
    if isinstance(answer, list):
        return ",".join(sorted(str(x).strip().casefold() for x in answer))
    return str(answer).strip().casefold()


def evaluate_qa(gold: list[dict], predictions: list[dict]) -> dict:
    """Gold/prediction arrays must use stable id and answer fields."""
    pred = {str(item["id"]): item["answer"] for item in predictions}
    by_type = defaultdict(lambda: {"count": 0, "correct": 0, "rouge_l_sum": 0.0})
    missing = []
    for item in gold:
        identifier = str(item["id"])
        if identifier not in pred:
            missing.append(identifier)
            continue
        stats = by_type[item.get("type", "unknown")]
        stats["count"] += 1
        stats["correct"] += int(_canonical_answer(item["answer"]) == _canonical_answer(pred[identifier]))
        stats["rouge_l_sum"] += rouge_l(_canonical_answer(item["answer"]),
                                        _canonical_answer(pred[identifier]))
    detailed = {kind: {"count": values["count"],
                       "accuracy": values["correct"] / values["count"],
                       "rouge_l": values["rouge_l_sum"] / values["count"]}
                for kind, values in by_type.items()}
    count = sum(v["count"] for v in by_type.values())
    return {"evaluated": count, "missing_ids": missing,
            "accuracy": sum(v["correct"] for v in by_type.values()) / count if count else 0.0,
            "rouge_l": sum(v["rouge_l_sum"] for v in by_type.values()) / count if count else 0.0,
            "by_type": detailed}
