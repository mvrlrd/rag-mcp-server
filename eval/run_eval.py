"""Прогон тестового набора вопросов (ANSWER_KEY.md) через реальный RAG-пайплайн.

CLI-скрипт, не pytest и не в CI — требует живой Ollama и наполненный индекс.
Запуск:

    python eval/run_eval.py
    python eval/run_eval.py --answer-key eval/answer_key.yaml --tag baseline

Пишет eval/report.json (для диффа между прогонами) и eval/report.md
(человекочитаемый отчёт со списком провалившихся вопросов). С флагом --tag
дополнительно сохраняет копию в eval/reports/<tag>_<дата>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_mcp import indexer
from rag_mcp.graph.build import ask_question

REFUSAL_MARKERS = ("информация отсутствует", "нет информации", "не найдено в")


def _looks_like_refusal(answer: str) -> bool:
    low = answer.lower()
    return any(marker in low for marker in REFUSAL_MARKERS)


def _keyword_match_count(answer: str, keywords: list[str]) -> int:
    low = answer.lower()
    return sum(1 for kw in keywords if kw.lower() in low)


def _source_match(sources: list[str], expected_sources: list[str]) -> bool:
    if not expected_sources:
        return True
    return any(s.endswith(exp) for s in sources for exp in expected_sources)


def run_question(q: dict, retries: int = 2) -> dict:
    start = time.monotonic()
    for attempt in range(retries + 1):
        try:
            out = ask_question(q["question"])
            break
        except Exception as e:
            if attempt == retries:
                raise
            print(f"  retry after error: {e}", file=sys.stderr)
    latency = time.monotonic() - start

    answer = out.get("answer", "")
    sources = out.get("sources", [])
    no_relevant_chunks = out.get("no_relevant_chunks", False)
    is_refusal = no_relevant_chunks or _looks_like_refusal(answer)

    refusal_expected = q.get("refusal_expected", False)
    if refusal_expected:
        correct = is_refusal
    else:
        matched = _keyword_match_count(answer, q.get("expected_keywords", []))
        correct = matched >= q.get("min_matches", 1) and not is_refusal

    return {
        "id": q["id"],
        "type": q["type"],
        "question": q["question"],
        "answer": answer,
        "sources": sources,
        "is_refusal": is_refusal,
        "refusal_expected": refusal_expected,
        "correct": correct,
        "source_match": _source_match(sources, q.get("expected_sources", [])),
        "has_expected_sources": bool(q.get("expected_sources")),
        "latency_s": latency,
    }


def compute_metrics(results: list[dict]) -> dict:
    total = len(results)
    correct = sum(r["correct"] for r in results)

    by_type: dict[str, dict] = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        bucket = by_type[r["type"]]
        bucket["total"] += 1
        bucket["correct"] += r["correct"]
    by_type_acc = {
        t: {"accuracy": b["correct"] / b["total"], "total": b["total"]}
        for t, b in by_type.items()
    }

    refusal_cases = [r for r in results if r["refusal_expected"]]
    predicted_refusals = [r for r in results if r["is_refusal"]]
    tp = sum(1 for r in predicted_refusals if r["refusal_expected"])
    refusal_precision = tp / len(predicted_refusals) if predicted_refusals else None
    refusal_recall = tp / len(refusal_cases) if refusal_cases else None

    with_sources = [r for r in results if r["has_expected_sources"]]
    source_match_rate = (
        sum(r["source_match"] for r in with_sources) / len(with_sources)
        if with_sources
        else None
    )

    avg_latency_s = sum(r["latency_s"] for r in results) / total if total else 0.0

    return {
        "accuracy": correct / total if total else 0.0,
        "total": total,
        "correct": correct,
        "by_type": by_type_acc,
        "refusal_precision": refusal_precision,
        "refusal_recall": refusal_recall,
        "source_match_rate": source_match_rate,
        "avg_latency_s": avg_latency_s,
    }


def write_report_md(path: Path, metrics: dict, results: list[dict]) -> None:
    acc_line = f"{metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})"
    lines = ["# Eval report", "", f"- accuracy: {acc_line}"]
    lines.append(f"- refusal_precision: {metrics['refusal_precision']}")
    lines.append(f"- refusal_recall: {metrics['refusal_recall']}")
    lines.append(f"- source_match_rate: {metrics['source_match_rate']}")
    lines.append(f"- avg_latency_s: {metrics['avg_latency_s']:.2f}")
    lines.append("")
    lines.append("## По типам")
    for t, b in sorted(metrics["by_type"].items()):
        lines.append(f"- {t}: {b['accuracy']:.2%} ({b['total']} вопросов)")
    lines.append("")
    failed = [r for r in results if not r["correct"]]
    lines.append(f"## Провалившиеся вопросы ({len(failed)})")
    for r in failed:
        lines.append(f"- [{r['id']}:{r['type']}] {r['question']}")
        lines.append(f"  - ответ: {r['answer'][:200]!r}")
        lines.append(f"  - источники: {r['sources']}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answer-key", default="eval/answer_key.yaml")
    parser.add_argument(
        "--tag", default=None, help="доп. копия в eval/reports/<tag>_<дата>.json"
    )
    args = parser.parse_args()

    status = indexer.index_status()
    if not status.get("chunks"):
        sys.exit("Индекс пуст. Сначала проиндексируй корпус (index_folder).")

    key_path = Path(args.answer_key)
    if not key_path.exists():
        sys.exit(
            f"Не найден {key_path}. Скопируй вопросы из ANSWER_KEY.md "
            "по схеме eval/answer_key.example.yaml."
        )
    questions = yaml.safe_load(key_path.read_text(encoding="utf-8"))

    results = [run_question(q) for q in questions]
    metrics = compute_metrics(results)

    report_path = Path("eval/report.json")
    payload = json.dumps(
        {"metrics": metrics, "results": results}, ensure_ascii=False, indent=2
    )
    report_path.write_text(payload, encoding="utf-8")
    write_report_md(Path("eval/report.md"), metrics, results)

    if args.tag:
        reports_dir = Path("eval/reports")
        reports_dir.mkdir(exist_ok=True)
        dated = f"{args.tag}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        (reports_dir / dated).write_text(payload, encoding="utf-8")

    acc = f"{metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})"
    print(f"accuracy: {acc}")
    print(f"refusal_precision: {metrics['refusal_precision']}")
    print(f"refusal_recall: {metrics['refusal_recall']}")
    print(f"source_match_rate: {metrics['source_match_rate']}")
    print("report: eval/report.json, eval/report.md")


if __name__ == "__main__":
    main()
