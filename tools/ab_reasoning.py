#!/usr/bin/env python3
"""A/B harness: does the multi-agent panel beat a single combined call?

For each task, produce two solutions — (A) one-shot combined call, (B) the
multi-agent panel (plan-vote → code → adversarial critique → repair) — then
have a blind LLM judge score both on an absolute rubric. Order is randomized
per task (recorded) to avoid position bias.

This isolates the value of the *orchestration*. By default both modes use the
same model (so any delta is the panel structure, not model choice); pass
distinct role models to also measure diversity.

Usage:
    python tools/ab_reasoning.py [--n 8] [--k 3] [--out results.json]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neo.adapters import OpenAIAdapter  # noqa: E402
from neo.config import NeoConfig  # noqa: E402
from neo.multi_agent import MultiAgentReasoner, _extract_json  # noqa: E402


class EffortAdapter:
    """Wrap an adapter to inject a fixed reasoning effort on every call — so
    both A/B arms run at the *same* effort (a fair, controlled comparison) and
    reasoning models don't burn the whole run on hidden tokens."""

    def __init__(self, inner, effort):
        self._inner = inner
        self._effort = effort

    def generate(self, messages, **kw):
        kw.setdefault("reasoning_effort", self._effort)
        return self._inner.generate(messages, **kw)

    def name(self):
        return getattr(self._inner, "name", lambda: "wrapped")()

# Self-contained tasks with real edge cases — the regime where an adversarial
# critic should actually catch things a one-shot pass misses.
TASKS = [
    "Write a Python function to merge two sorted linked lists into one sorted list.",
    "Deduplicate a list of dicts while preserving first-seen order.",
    "Parse a semver string (with optional pre-release) and compare two versions.",
    "Implement an LRU cache with O(1) get and put.",
    "Return the longest palindromic substring of a string.",
    "Implement a token-bucket rate limiter (Python class).",
    "Write a `debounce(wait)` decorator that delays calls until idle.",
    "Validate that a string has balanced (), [], and {} brackets.",
    "Chunk an iterable into lists of size n, including a final short chunk.",
    "Flatten an arbitrarily nested list of integers (no recursion depth blowup).",
]

_SINGLE_SYS = (
    "You are a senior engineer. Solve the task in ONE shot. Respond ONLY with JSON: "
    '{"code": str, "explanation": str, "edge_cases": [str], "confidence": 0.0-1.0}'
)

_JUDGE_SYS = (
    "You are a strict code reviewer judging two solutions to the same task. "
    "Score each 1-10 on: correctness first, then edge-case handling and completeness. "
    "Be skeptical; penalize bugs and missed edge cases heavily. Ignore verbosity. "
    'Respond ONLY with JSON: {"score_a": int, "score_b": int, '
    '"winner": "A"|"B"|"tie", "reason": str}'
)


def single_call(adapter, task: str) -> dict:
    raw = adapter.generate(
        [{"role": "system", "content": _SINGLE_SYS}, {"role": "user", "content": task}],
        max_tokens=8000, temperature=0.3,
    )
    return _extract_json(raw) or {"code": raw, "explanation": "", "edge_cases": []}


def judge(adapter, task: str, sol_a: str, sol_b: str) -> dict:
    user = f"TASK:\n{task}\n\n--- SOLUTION A ---\n{sol_a}\n\n--- SOLUTION B ---\n{sol_b}"
    raw = adapter.generate(
        [{"role": "system", "content": _JUDGE_SYS}, {"role": "user", "content": user}],
        max_tokens=4000, temperature=0.0,
    )
    obj = _extract_json(raw) or {}
    return {
        "score_a": int(obj.get("score_a", 0) or 0),
        "score_b": int(obj.get("score_b", 0) or 0),
        "winner": obj.get("winner", "tie"),
        "reason": obj.get("reason", ""),
    }


def run(n: int, k: int, model: str, out: Path, effort: str = "low") -> dict:
    key = NeoConfig.load().api_key
    adapter = EffortAdapter(OpenAIAdapter(model=model, api_key=key), effort)
    reasoner = MultiAgentReasoner(lambda role: adapter, k_plans=k, max_repair_rounds=1)

    rng = random.Random(1234)
    tasks = TASKS[:n]
    rows = []
    single_wins = multi_wins = ties = 0
    single_score_sum = multi_score_sum = 0
    single_time = multi_time = 0.0

    for i, task in enumerate(tasks, 1):
        try:
            t0 = time.time()
            s = single_call(adapter, task)
            single_time += time.time() - t0
            single_sol = s.get("code", "")

            t0 = time.time()
            d = reasoner.deliberate(task)
            multi_time += time.time() - t0
            multi_sol = d.code_suggestions[0].code_block if d.code_suggestions else ""
        except Exception as e:  # keep the run going; record the failure
            print(f"[{i}/{len(tasks)}] ERROR: {type(e).__name__}: {str(e)[:120]}", flush=True)
            rows.append({"task": task, "error": f"{type(e).__name__}: {str(e)[:200]}"})
            continue

        # Randomize which slot (A/B) is the multi-agent solution.
        multi_is_a = rng.random() < 0.5
        sol_a, sol_b = (multi_sol, single_sol) if multi_is_a else (single_sol, multi_sol)
        v = judge(adapter, task, sol_a, sol_b)

        multi_score = v["score_a"] if multi_is_a else v["score_b"]
        single_score = v["score_b"] if multi_is_a else v["score_a"]
        winner = v["winner"]
        if winner == "tie":
            multi_won = None
            ties += 1
        else:
            won_a = winner == "A"
            multi_won = (won_a and multi_is_a) or (not won_a and not multi_is_a)
            if multi_won:
                multi_wins += 1
            else:
                single_wins += 1

        multi_score_sum += multi_score
        single_score_sum += single_score
        rows.append({
            "task": task, "multi_is_a": multi_is_a, "winner": winner,
            "multi_won": multi_won, "multi_score": multi_score, "single_score": single_score,
            "multi_confidence": round(d.confidence, 3), "multi_rounds": d.rounds,
            "reason": v["reason"][:200],
        })
        print(f"[{i}/{len(tasks)}] multi={multi_score} single={single_score} "
              f"winner={'multi' if multi_won else ('single' if multi_won is False else 'tie')} "
              f":: {task[:50]}", flush=True)

    scored = multi_wins + single_wins + ties
    denom = max(1, scored)
    summary = {
        "model": model, "effort": effort, "n": len(tasks), "scored": scored, "k_plans": k,
        "multi_wins": multi_wins, "single_wins": single_wins, "ties": ties,
        "multi_win_rate": round(multi_wins / denom, 3),
        "decisive_multi_win_rate": round(multi_wins / max(1, multi_wins + single_wins), 3),
        "avg_multi_score": round(multi_score_sum / denom, 2),
        "avg_single_score": round(single_score_sum / denom, 2),
        "avg_score_delta": round((multi_score_sum - single_score_sum) / denom, 2),
        "avg_single_time_s": round(single_time / denom, 1),
        "avg_multi_time_s": round(multi_time / denom, 1),
    }
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--out", default="/tmp/ab_reasoning_results.json")
    a = ap.parse_args()
    run(a.n, a.k, a.model, Path(a.out), effort=a.effort)
