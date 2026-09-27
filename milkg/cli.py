"""Command-line pipeline without GraphGen's WebUI or heavyweight services."""

import argparse
import json
import os
from pathlib import Path

from .documents import load_chunks
from .extraction import extract_chunk
from .graph import MilitaryGraph
from .llm import OpenAICompatibleModel
from .qa import QUESTION_TYPES, export_sft, generate_qa
from .traversal import traverse, atomic_facts
from .training import evaluate_extractions, prepare_extractor_sft, read_json_array
from .evaluation import evaluate_qa, random_edge_baseline, traversal_metrics
from .augmentation import materialize_specifications
from .documents import Chunk


def _model(args, role: str) -> OpenAICompatibleModel:
    url = getattr(args, f"{role}_url", None) or os.getenv(f"MILKG_{role.upper()}_URL")
    name = getattr(args, f"{role}_model", None) or os.getenv(f"MILKG_{role.upper()}_MODEL")
    key = getattr(args, f"{role}_key", None) or os.getenv(f"MILKG_{role.upper()}_KEY", "")
    if not url or not name:
        raise ValueError(f"Set --{role}-url and --{role}-model or MILKG_{role.upper()}_URL/MODEL")
    return OpenAICompatibleModel(url, name, key, json_mode=not args.no_json_mode,
                                 enable_thinking=False if args.disable_thinking else None)


def _synonyms(path: Path | None) -> dict[str, str]:
    return json.loads(path.read_text(encoding="utf-8")) if path else {}


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _build(args, output: Path) -> MilitaryGraph:
    if args.extractions:
        extractions = json.loads(args.extractions.read_text(encoding="utf-8"))
        if not isinstance(extractions, list):
            raise ValueError("--extractions must contain a JSON array")
    else:
        if not args.inputs:
            raise ValueError("Provide input documents or --extractions")
        model = _model(args, "extract")
        chunks = load_chunks(args.inputs, args.max_chars, args.overlap)
        extractions = [extract_chunk(chunk, model, args.min_confidence,
                                     args.extract_temperature) for chunk in chunks]
        _write_json(output / "chunks.json", [chunk.__dict__ for chunk in chunks])
    _write_json(output / "extractions.json", extractions)
    kg = MilitaryGraph(_synonyms(args.synonyms), args.fuzzy_threshold)
    for extraction in extractions:
        kg.add_extraction(extraction)
    kg.save(output / "graph.json")
    return kg


def _generate(args, output: Path, kg: MilitaryGraph) -> None:
    subgraphs = traverse(kg, min_depth=args.min_depth, max_depth=args.max_depth,
                         max_paths=args.max_paths, max_subgraphs=args.max_subgraphs,
                         max_overlap=args.max_overlap)
    if args.include_atomic:
        subgraphs.extend(atomic_facts(kg)[:args.max_atomic])
    _write_json(output / "subgraphs.json", [s.to_dict() for s in subgraphs])
    if not subgraphs:
        _write_json(output / "qa.json", [])
        _write_json(output / "stats.json", {"attempted": 0, "accepted": 0, "reason": "No subgraphs"})
        export_sft([], output / f"sft_{args.format}.json", args.format)
        return
    model = _model(args, "generate")
    types = tuple(x.strip() for x in args.question_types.split(",") if x.strip())
    if not types or not set(types) <= set(QUESTION_TYPES):
        raise ValueError(f"Question types must be among: {QUESTION_TYPES}")
    previous_items = read_json_array(output / "qa.json") if args.resume and (output / "qa.json").exists() else None
    previous_stats = json.loads((output / "stats.json").read_text(encoding="utf-8")) if args.resume and (output / "stats.json").exists() else None
    def checkpoint(items: list[dict], stats: dict) -> None:
        _write_json(output / "qa.json", items)
        _write_json(output / "stats.json", stats)
        export_sft(items, output / f"sft_{args.format}.json", args.format)
    items, stats = generate_qa(kg, subgraphs, model, types, args.seed, args.per_subgraph,
                               args.dedup_threshold, previous_items, previous_stats, checkpoint,
                               args.generate_temperature)
    checkpoint(items, stats)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MilKG-QA paper reproduction")
    parser.add_argument("command", choices=("build", "traverse", "generate", "run", "augment-specs",
                                            "prepare-extractor", "evaluate-extractor",
                                            "evaluate-traversal", "evaluate-qa"))
    parser.add_argument("inputs", nargs="*", type=Path, help="TXT, MD, PDF, DOCX files or directories")
    parser.add_argument("--output", type=Path, default=Path("milkg_output"))
    parser.add_argument("--graph", type=Path, help="existing graph.json for traverse/generate")
    parser.add_argument("--extractions", type=Path, help="trusted pre-extracted JSON array for offline build")
    parser.add_argument("--annotations", type=Path, help="gold annotation JSON array")
    parser.add_argument("--predictions", type=Path, help="prediction JSON array for F1 evaluation")
    parser.add_argument("--synonyms", type=Path, help="JSON alias-to-canonical dictionary")
    parser.add_argument("--extract-url")
    parser.add_argument("--extract-model")
    parser.add_argument("--extract-key")
    parser.add_argument("--generate-url")
    parser.add_argument("--generate-model")
    parser.add_argument("--generate-key")
    parser.add_argument("--no-json-mode", action="store_true", help="omit response_format for endpoints that reject JSON mode")
    parser.add_argument("--disable-thinking", action="store_true", help="disable Qwen thinking for reliable JSON output")
    parser.add_argument("--max-chars", type=int, default=1800)
    parser.add_argument("--overlap", type=int, default=180)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--extract-temperature", type=float, default=0.1)
    parser.add_argument("--generate-temperature", type=float, default=0.7)
    parser.add_argument("--fuzzy-threshold", type=float, default=0.9)
    parser.add_argument("--min-depth", type=int, default=2)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--max-paths", type=int, default=2000)
    parser.add_argument("--max-subgraphs", type=int, default=500)
    parser.add_argument("--include-atomic", action="store_true", help="also generate easy one-hop fact questions")
    parser.add_argument("--max-atomic", type=int, default=30)
    parser.add_argument("--max-overlap", type=float, default=0.5)
    parser.add_argument("--question-types", default=",".join(QUESTION_TYPES))
    parser.add_argument("--per-subgraph", type=int, default=1)
    parser.add_argument("--dedup-threshold", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--format", choices=("alpaca", "sharegpt", "chatml"), default="alpaca")
    parser.add_argument("--resume", action="store_true", help="continue from qa.json/stats.json checkpoints")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == "prepare-extractor":
        if not args.annotations:
            parser.error("prepare-extractor requires --annotations")
        _write_json(args.output / "extractor_sft_alpaca.json",
                    prepare_extractor_sft(read_json_array(args.annotations)))
        print(f"MilKG-QA output: {args.output.resolve()}")
        return 0
    if args.command == "evaluate-extractor":
        if not args.annotations or not args.predictions:
            parser.error("evaluate-extractor requires --annotations and --predictions")
        _write_json(args.output / "extraction_metrics.json",
                    evaluate_extractions(read_json_array(args.annotations),
                                         read_json_array(args.predictions)))
        print(f"MilKG-QA output: {args.output.resolve()}")
        return 0
    if args.command == "evaluate-qa":
        if not args.annotations or not args.predictions:
            parser.error("evaluate-qa requires --annotations and --predictions")
        _write_json(args.output / "qa_metrics.json",
                    evaluate_qa(read_json_array(args.annotations),
                                read_json_array(args.predictions)))
        print(f"MilKG-QA output: {args.output.resolve()}")
        return 0
    if args.command == "augment-specs":
        graph_path = args.graph or args.output / "graph.json"
        kg = MilitaryGraph.load(graph_path, _synonyms(args.synonyms))
        chunks_path = args.output / "chunks.json"
        chunks = [Chunk(**item) for item in read_json_array(chunks_path)]
        added = materialize_specifications(kg, chunks)
        kg.save(args.output / "graph_augmented.json")
        print(f"Materialized {added} source-grounded Weapon-Spec edges")
        print(f"MilKG-QA output: {args.output.resolve()}")
        return 0
    if args.command in {"build", "run"}:
        kg = _build(args, args.output)
    else:
        graph_path = args.graph or args.output / "graph.json"
        kg = MilitaryGraph.load(graph_path, _synonyms(args.synonyms))
    if args.command in {"traverse", "evaluate-traversal"}:
        subgraphs = traverse(kg, min_depth=args.min_depth, max_depth=args.max_depth,
                             max_paths=args.max_paths, max_subgraphs=args.max_subgraphs,
                             max_overlap=args.max_overlap)
        _write_json(args.output / "subgraphs.json", [s.to_dict() for s in subgraphs])
        if args.command == "evaluate-traversal":
            baseline = random_edge_baseline(kg, subgraphs, args.seed)
            _write_json(args.output / "traversal_metrics.json",
                        {"milkg": traversal_metrics(kg, subgraphs),
                         "random_edge_baseline": traversal_metrics(kg, baseline)})
    elif args.command in {"generate", "run"}:
        _generate(args, args.output, kg)
    print(f"MilKG-QA output: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
