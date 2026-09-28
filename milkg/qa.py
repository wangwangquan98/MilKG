"""Five MilKG-QA question types, grounded validation, deduplication and export."""

import json
import random
import re
from pathlib import Path
from typing import Callable

from .graph import MilitaryGraph, normalize_name
from .llm import ChatModel
from .traversal import Subgraph


QUESTION_TYPES = ("single_choice", "multiple_choice", "cot", "true_false", "fill_blank")
STYLES = ("formal_exam", "scenario_analysis", "decision_making")

GENERATION_SYSTEM = """你是军事知识图谱题目生成器。只根据提供的子图事实出题，不使用外部常识。
输出单个 JSON 对象，字段：question、options、answer、explanation、supporting_facts、referenced_entities、answer_entities。
options 是四个 {"label":"A/B/C/D","text":"..."} 对象，非选择题为 []。
选择题 answer 为正确选项标签数组；其他题 answer 为字符串。
supporting_facts 是所用边 ID 数组；referenced_entities 是问题和答案涉及的实体 ID 数组；
answer_entities 是答案所指实体 ID 数组。所有 ID 必须来自输入。CoT 必须逐步解释至少两条事实。
真假题 answer 只能是“正确”或“错误”；当要求错误题时，还必须输出
altered_fact={"edge_id":原边ID,"replacement_node_id":替换目标实体ID}，将原边目标换成同类型、不同且图中无该关系的实体。
题干和解释不要引入子图外的事实；干扰选项可用给出的同类型已知实体，不得与正确选项等价。
question、options、answer、explanation 将直接用于训练没有图谱输入的模型，必须独立可读。
解释要直接写出所依据的人物、装备、关系和必要的数值，不能写“根据子图/图谱/给定资料/事实编号”、
“图中没有证据”等依赖外部上下文的话，也不能出现 e000001、n000001 等内部 ID。
内部 ID 只允许出现在 supporting_facts、referenced_entities、answer_entities、altered_fact 字段。
选择题的解释只说明正确选项成立的正面事实，不评论错误选项；不能把“子图未收录”当作错误依据。"""

REWRITE_SYSTEM = """把候选问答改写成可直接用于训练的独立问答。只使用给出的事实，不补充常识。
只返回 JSON 对象，包含 question、options、answer、explanation 四个字段；题型、正确答案和选项含义不变。
题干、选项、答案、解释必须自足：直接陈述具体事实，不能提子图、图谱、资料、事实编号、实体编号、
内部 ID 或“图中未证实”。选择题只解释正确选项，不提错误选项；其他题也只陈述答案所需的正面事实。"""

_INTERNAL_ID = re.compile(r"(?<![A-Za-z0-9_])[en]\d{4,}(?![A-Za-z0-9_])", re.I)
_EXTERNAL_CONTEXT = re.compile(
    r"子图|知识图谱|图谱|(?:给定|所给|提供|上述)(?:的)?(?:资料|材料|信息)|"
    r"(?:给定|所给|提供)(?:的)?事实|事实编号|实体编号|边编号|"
    r"(?:实体|边|事实)\s*ID|属性信息|未被证实|(?:图|表|文)中", re.I
)


def training_text_issue(item: dict) -> str | None:
    """Check only fields visible to the future model, not provenance fields."""
    fields = [item.get("question"), item.get("explanation"), item.get("answer")]
    options = item.get("options") or []
    if not isinstance(options, list):
        return "invalid options"
    fields.extend(option.get("text") for option in options if isinstance(option, dict))
    for value in fields:
        values = value if isinstance(value, list) else [value]
        for part in values:
            if isinstance(part, str) and (_INTERNAL_ID.search(part) or _EXTERNAL_CONTEXT.search(part)):
                return "graph reference in training text"
    return None


def difficulty(depth: int) -> str:
    return "easy" if depth <= 1 else "medium" if depth == 2 else "hard"


def subgraph_payload(kg: MilitaryGraph, subgraph: Subgraph) -> dict:
    return {"strategy": subgraph.strategy, "depth": subgraph.depth,
            "nodes": [{"id": n, **kg.graph.nodes[n]} for n in subgraph.nodes],
            "facts": [{"id": e, "source": kg.edge(e)[0], "target": kg.edge(e)[1], **kg.edge(e)[2]}
                      for e in subgraph.edges]}


def generation_prompt(kg: MilitaryGraph, subgraph: Subgraph, question_type: str,
                      style: str, truth: bool = True) -> str:
    instructions = {
        "single_choice": "四选一：恰好1个正确、3个同类但错误的干扰项。",
        "multiple_choice": "四选多：恰好2或3个正确，其他为同类干扰项。",
        "cot": "简答推理：必须通过至少2条事实给出分步推理。",
        "true_false": f"判断题：生成{'正确' if truth else '错误'}陈述；错误题只改动一处目标实体。",
        "fill_blank": "填空题：用____遮住一个有唯一答案的实体或技术参数。",
    }
    relevant_types = {kg.graph.nodes[node]["type"] for node in subgraph.nodes}
    counts = {kind: 0 for kind in relevant_types}
    distractors = []
    for node, data in kg.graph.nodes(data=True):
        kind = data["type"]
        if kind in relevant_types and (counts[kind] < 16 or node in subgraph.nodes):
            distractors.append({"id": node, "name": data["name"], "type": kind})
            counts[kind] += 1
    return (f"题型：{question_type}。风格：{style}。难度：{difficulty(subgraph.depth)}。"
            f"要求：{instructions[question_type]}\n子图：\n"
            + json.dumps(subgraph_payload(kg, subgraph), ensure_ascii=False)
            + "\n可用于选项的已知实体（同类型，选项文本只写实体名）：\n"
            + json.dumps(distractors, ensure_ascii=False))


def _names(kg: MilitaryGraph, node: str) -> list[str]:
    data = kg.graph.nodes[node]
    return [data["name"], *data.get("aliases", [])]


def _has_fact(kg: MilitaryGraph, source: str, target: str, kind: str) -> bool:
    return any(data["type"] == kind for data in kg.graph.get_edge_data(source, target, default={}).values())


def validate_qa(raw: dict, kg: MilitaryGraph, subgraph: Subgraph,
                question_type: str, expected_truth: bool = True) -> tuple[bool, str]:
    if not isinstance(raw, dict):
        return False, "not an object"
    question, answer = raw.get("question"), raw.get("answer")
    if not isinstance(question, str) or not question.strip():
        return False, "missing question"
    if not isinstance(raw.get("explanation"), str) or not raw["explanation"].strip():
        return False, "missing explanation"
    issue = training_text_issue(raw)
    if issue:
        return False, issue
    if len(question + str(answer)) < 10 or len(question + str(answer)) > 2000:
        return False, "length outside [10,2000]"
    supporting = raw.get("supporting_facts")
    referenced = raw.get("referenced_entities")
    answer_entities = raw.get("answer_entities")
    if not isinstance(supporting, list) or not supporting or not set(supporting) <= set(subgraph.edges):
        return False, "unverifiable supporting facts"
    if not isinstance(referenced, list) or not referenced or not set(referenced) <= set(kg.graph.nodes):
        return False, "unknown referenced entity"
    if not set(referenced) & set(subgraph.nodes):
        return False, "no source-subgraph entity referenced"
    if not isinstance(answer_entities, list) or not set(answer_entities) <= set(subgraph.nodes):
        return False, "unknown answer entity"
    # Declared entities must actually be visible in the item. This catches
    # invented IDs, though free-form hallucination needs human/model review.
    visible = question + " " + str(answer) + " " + str(raw.get("explanation", "")) + " " + str(raw.get("options", []))
    if not any(any(normalize_name(name) in normalize_name(visible) for name in _names(kg, node))
               for node in referenced):
        return False, "referenced entity not present in text"
    options = raw.get("options", [])
    if question_type in {"single_choice", "multiple_choice"}:
        if not isinstance(options, list) or len(options) != 4:
            return False, "choice question needs four options"
        if any(not isinstance(o, dict) or not isinstance(o.get("text"), str) for o in options):
            return False, "invalid option"
        labels = [o.get("label") for o in options]
        texts = [o["text"].strip() for o in options]
        if sorted(labels) != ["A", "B", "C", "D"] or len({normalize_name(t) for t in texts}) != 4 or any(not t for t in texts):
            return False, "non-distinct options"
        if not isinstance(answer, list) or not set(answer) <= set(labels):
            return False, "invalid choice answer"
        if len(answer) != (1 if question_type == "single_choice" else len(answer)):
            return False, "wrong number of answers"
        if question_type == "multiple_choice" and len(answer) not in {2, 3}:
            return False, "multi-choice needs 2-3 answers"
        correct_text = " ".join(o["text"] for o in options if o["label"] in answer)
        if not answer_entities or any(not any(normalize_name(name) in normalize_name(correct_text)
                                              for name in _names(kg, node))
                                      for node in answer_entities):
            return False, "correct options not grounded in answer entities"
        answer_types = {kg.graph.nodes[node]["type"] for node in answer_entities}
        for option in options:
            candidates = [node for node in kg.graph if any(normalize_name(option["text"]) == normalize_name(name)
                                                            for name in _names(kg, node))]
            if not candidates or not any(kg.graph.nodes[node]["type"] in answer_types for node in candidates):
                return False, "option is not a same-type KG entity"
            if option["label"] in answer and not any(node in answer_entities for node in candidates):
                return False, "correct option does not name answer entity"
            if option["label"] not in answer and any(node in answer_entities for node in candidates):
                return False, "distractor names an answer entity"
        for option in options:
            if option["label"] not in answer and any(normalize_name(name) == normalize_name(option["text"])
                                                       for node in answer_entities
                                                       for name in _names(kg, node)):
                return False, "distractor equals correct entity"
        explanation = normalize_name(raw["explanation"])
        if any(normalize_name(option["text"]) in explanation
               for option in options if option["label"] not in answer):
            return False, "explanation discusses distractor"
    else:
        if options not in ([], None):
            return False, "non-choice question has options"
        if not isinstance(answer, str) or not answer.strip():
            return False, "missing text answer"
        if question_type == "cot":
            if len(supporting) < 2 or len(str(raw.get("explanation", ""))) < 15:
                return False, "CoT lacks multiple evidence steps"
        elif question_type == "fill_blank":
            if "____" not in question or not answer_entities:
                return False, "fill blank lacks mask or entity answer"
            if not any(normalize_name(name) == normalize_name(answer)
                       for node in answer_entities for name in _names(kg, node)):
                # Technical attribute values may also be the masked answer.
                if not any(attr["value"] == answer for node in answer_entities
                           for attr in kg.graph.nodes[node].get("attributes", {}).values()):
                    return False, "fill answer absent from graph"
        elif question_type == "true_false":
            if answer != ("正确" if expected_truth else "错误"):
                return False, "wrong truth label"
            if not expected_truth:
                altered = raw.get("altered_fact")
                if not isinstance(altered, dict) or altered.get("edge_id") not in supporting:
                    return False, "missing altered fact"
                edge = kg.edge(altered["edge_id"])
                replacement = altered.get("replacement_node_id")
                if not edge or replacement not in kg.graph or replacement == edge[1]:
                    return False, "invalid false-fact replacement"
                if kg.graph.nodes[replacement]["type"] != kg.graph.nodes[edge[1]]["type"]:
                    return False, "false-fact replacement has wrong entity type"
                if _has_fact(kg, edge[0], replacement, edge[2]["type"]):
                    return False, "altered fact is present in graph"
                if not any(normalize_name(name) in normalize_name(question) for name in _names(kg, replacement)):
                    return False, "altered entity not in question"
    return True, "ok"


def _lcs_length(a: list[str], b: list[str]) -> int:
    previous = [0] * (len(b) + 1)
    for char in a:
        current = [0]
        for i, other in enumerate(b, 1):
            current.append(previous[i - 1] + 1 if char == other else max(previous[i], current[-1]))
        previous = current
    return previous[-1]


def rouge_l(a: str, b: str) -> float:
    # Character-level ROUGE-L works for Chinese without a tokenizer.
    left = list(re.sub(r"\s+", "", a))
    right = list(re.sub(r"\s+", "", b))
    if not left or not right:
        return 0.0
    lcs = _lcs_length(left, right)
    return 2 * lcs / (len(left) + len(right))


def generate_qa(kg: MilitaryGraph, subgraphs: list[Subgraph], model: ChatModel,
                question_types: tuple[str, ...] = QUESTION_TYPES, seed: int = 42,
                per_subgraph: int = 1, dedup_threshold: float = 0.85,
                existing_items: list[dict] | None = None, existing_stats: dict | None = None,
                on_progress: Callable[[list[dict], dict], None] | None = None,
                temperature: float = 0.7) -> tuple[list[dict], dict]:
    unknown = set(question_types) - set(QUESTION_TYPES)
    if unknown or not question_types:
        raise ValueError(f"Unknown question types: {sorted(unknown)}")
    if per_subgraph < 1:
        raise ValueError("per_subgraph must be >= 1")
    rng = random.Random(seed)
    accepted: list[dict] = list(existing_items or [])
    stats: dict[str, int] = dict(existing_stats or {"attempted": 0, "accepted": 0, "duplicates": 0, "invalid": 0})
    completed = stats["attempted"]
    # Cycle types so small runs still cover all five; alternate truth values.
    for i, subgraph in enumerate(subgraphs):
        for j in range(per_subgraph):
            attempt_index = i * per_subgraph + j
            style = rng.choice(STYLES)
            if attempt_index < completed:
                continue
            question_type = question_types[(i * per_subgraph + j) % len(question_types)]
            if subgraph.depth == 1 and question_type in {"cot", "multiple_choice"}:
                simple = tuple(t for t in question_types if t in {"single_choice", "true_false", "fill_blank"})
                if simple:
                    question_type = simple[attempt_index % len(simple)]
            truth = (stats["attempted"] % 2 == 0)
            stats["attempted"] += 1
            try:
                raw = model.complete(GENERATION_SYSTEM,
                                     generation_prompt(kg, subgraph, question_type, style, truth), temperature)
            except (RuntimeError, ValueError, TimeoutError) as exc:
                stats["invalid"] += 1
                reason = f"model_error:{type(exc).__name__}"
                stats[reason] = stats.get(reason, 0) + 1
                if on_progress:
                    on_progress(accepted, stats)
                continue
            valid, reason = validate_qa(raw, kg, subgraph, question_type, truth)
            if not valid and reason in {"graph reference in training text",
                                        "explanation discusses distractor"}:
                try:
                    rewrite_prompt = (generation_prompt(kg, subgraph, question_type, style, truth)
                                      + "\n待改写候选：\n"
                                      + json.dumps(raw, ensure_ascii=False))
                    rewritten = model.complete(REWRITE_SYSTEM, rewrite_prompt, temperature)
                    if isinstance(rewritten, dict):
                        candidate = {**raw, **{key: rewritten[key]
                                              for key in ("question", "explanation")
                                              if key in rewritten}}
                        if training_text_issue(candidate):
                            candidate.update({key: rewritten[key]
                                              for key in ("options", "answer") if key in rewritten})
                        valid, reason = validate_qa(candidate, kg, subgraph, question_type, truth)
                        if valid:
                            raw = candidate
                            stats["rewritten"] = stats.get("rewritten", 0) + 1
                except (RuntimeError, ValueError, TimeoutError):
                    pass
            if not valid:
                stats["invalid"] += 1
                stats[f"invalid:{reason}"] = stats.get(f"invalid:{reason}", 0) + 1
                if on_progress:
                    on_progress(accepted, stats)
                continue
            if any(rouge_l(raw["question"], item["question"]) >= dedup_threshold for item in accepted):
                stats["duplicates"] += 1
                if on_progress:
                    on_progress(accepted, stats)
                continue
            item = {**raw, "type": question_type, "style": style,
                    "difficulty": difficulty(subgraph.depth), "strategy": subgraph.strategy,
                    "subgraph_edges": list(subgraph.edges)}
            accepted.append(item)
            stats["accepted"] += 1
            if on_progress:
                on_progress(accepted, stats)
    return accepted, stats


def _response(item: dict) -> str:
    answer = item["answer"]
    if isinstance(answer, list):
        answer = ",".join(sorted(answer))
    explanation = item.get("explanation", "")
    return f"{answer}\n{explanation}".strip()


def export_sft(items: list[dict], path: Path, fmt: str = "alpaca") -> None:
    if fmt not in {"alpaca", "sharegpt", "chatml"}:
        raise ValueError(f"Unknown output format: {fmt}")
    result = []
    for item in items:
        issue = training_text_issue(item)
        if issue:
            raise ValueError(f"Cannot export SFT item with {issue}: {item.get('question', '')[:80]}")
        options = item.get("options") or []
        instruction = item["question"]
        if options:
            instruction += "\n" + "\n".join(f"{o['label']}. {o['text']}" for o in options)
        response = _response(item)
        if fmt == "alpaca":
            record = {"instruction": instruction, "input": "", "output": response}
        elif fmt == "sharegpt":
            record = {"conversations": [{"from": "human", "value": instruction},
                                        {"from": "gpt", "value": response}]}
        else:
            record = {"messages": [{"role": "user", "content": instruction},
                                   {"role": "assistant", "content": response}]}
        result.append(record)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
