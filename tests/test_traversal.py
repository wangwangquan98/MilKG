import pytest

from milkg.graph import MilitaryGraph
from milkg.llm import RequestCancelled
from milkg.traversal import traverse


def specification_graph(count):
    kg = MilitaryGraph()
    for index in range(count):
        weapon, specification = f"w{index:03d}", f"s{index:03d}"
        kg.graph.add_node(weapon, type="Weapon System")
        kg.graph.add_node(specification, type="Technical Specification")
        edge = f"e{index:03d}"
        kg.graph.add_edge(weapon, specification, key=edge, id=edge,
                          type="Weapon-Spec", confidence=0.4 + index / (count * 2))
    kg.graph.add_node("isolated", type="Military Personnel")
    return kg


def test_ranked_comparisons_keep_overlap_boundaries_and_scan_edges_once(monkeypatch):
    kg = specification_graph(12)
    original_edges = kg.edges
    scans = []

    def counted_edges(*args):
        scans.append(1)
        yield from original_edges(*args)

    monkeypatch.setattr(kg, "edges", counted_edges)
    selected = traverse(kg, max_subgraphs=100, max_overlap=0.5)
    assert len(scans) == 1
    assert [subgraph.edges for subgraph in selected] == [
        (f"e{left:03d}", f"e{left + 1:03d}") for left in range(10, -1, -2)
    ]
    assert all(subgraph.strategy == "entity_comparison" for subgraph in selected)
    # With overlap 1, all distinct pairs remain; with overlap 0 only the
    # reserved comparison remains, including when pairs are disjoint.
    assert len(traverse(kg, max_subgraphs=100, max_overlap=1)) == 66
    assert len(traverse(kg, max_subgraphs=100, max_overlap=0)) == 1
    assert len(traverse(kg, max_subgraphs=2, max_overlap=0.5)) == 2


def test_traversal_reports_candidate_progress_and_can_be_cancelled():
    kg = specification_graph(145)
    messages = []

    def progress(message):
        messages.append(message)
        if "已评分 10000" in message:
            raise RequestCancelled()

    with pytest.raises(RequestCancelled):
        traverse(kg, max_subgraphs=1000, on_progress=progress)
    assert any("正在遍历实体比较" in message for message in messages)
    assert not any("子图筛选完成" in message for message in messages)
