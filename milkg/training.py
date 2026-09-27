"""Prepare extractor SFT labels and compute paper-style exact entity/relation F1."""

import json
from pathlib import Path

from .documents import Chunk
from .extraction import EXTRACTION_SYSTEM, extraction_prompt, validate_extraction
from .graph import normalize_name


def prepare_extractor_sft(annotations: list[dict]) -> list[dict]:
    """Annotations: [{id, source, text, entities, relations}, ...]."""
    records = []
    for i, item in enumerate(annotations):
        if not isinstance(item.get("text"), str):
            raise ValueError(f"Annotation {i} lacks text")
        chunk = Chunk(str(item.get("id", i)), str(item.get("source", "annotation")), item["text"])
        target = validate_extraction(item, chunk)
        entities = [{k: value for k, value in entity.items() if k != "chunk_id"}
                    for entity in target["entities"]]
        relations = [{k: value for k, value in relation.items() if k != "chunk_id"}
                     for relation in target["relations"]]
        records.append({"instruction": EXTRACTION_SYSTEM + "\n" + extraction_prompt(chunk),
                        "input": "", "output": json.dumps({"entities": entities,
                                                            "relations": relations}, ensure_ascii=False)})
    return records


def _sets(item: dict) -> tuple[set[tuple], set[tuple]]:
    entities = {(normalize_name(e["name"]), e["type"]) for e in item.get("entities", [])}
    relations = {(normalize_name(r["source"]), r["type"], normalize_name(r["target"]))
                 for r in item.get("relations", [])}
    return entities, relations


def _f1(gold: list[set], predicted: list[set]) -> dict[str, float | int]:
    tp = sum(len(a & b) for a, b in zip(gold, predicted))
    fp = sum(len(b - a) for a, b in zip(gold, predicted))
    fn = sum(len(a - b) for a, b in zip(gold, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "tp": tp, "fp": fp, "fn": fn}


def evaluate_extractions(gold: list[dict], predictions: list[dict]) -> dict:
    if len(gold) != len(predictions):
        raise ValueError("Gold and prediction files must have the same number of chunks")
    gold_sets = [_sets(item) for item in gold]
    prediction_sets = [_sets(item) for item in predictions]
    return {"chunks": len(gold),
            "entity": _f1([x[0] for x in gold_sets], [x[0] for x in prediction_sets]),
            "relation": _f1([x[1] for x in gold_sets], [x[1] for x in prediction_sets])}


def read_json_array(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected JSON array: {path}")
    return value
