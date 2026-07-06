#!/usr/bin/env python3
"""Controlled A/B/A: isolate model *diversity* from orchestration.

Three arms produced for the SAME task, scored in ONE judge call (order
randomized), so there's no cross-run judge variance:

    A  single  = one combined gpt-5.5 call
    B  panel_gpt    = panel with a SAME-model critic (gpt-5.5)
    C  panel_claude = panel with a DISTINCT critic (Claude)

Therefore:
    orchestration gain = B - A   (structure, no diversity)
    diversity gain     = C - B   (only the critic's model changed)

Usage:
    python tools/ab_controlled.py [--n 8] [--k 2] [--out results.json]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for ab_reasoning

import ab_reasoning as ab  # noqa: E402  (sets up src path + shared helpers)
from neo.adapters import OpenAIAdapter  # noqa: E402
from neo.config import NeoConfig  # noqa: E402
from neo.multi_agent import MultiAgentReasoner, _extract_json  # noqa: E402

_JUDGE3_SYS = (
    "You are a strict code reviewer judging three solutions to the same task. "
    "Score EACH 1-10 on correctness first, then edge-case handling and completeness. "
    "Be skeptical; penalize bugs and missed edge cases heavily. Ignore verbosity. "
    'Respond ONLY with JSON: {"score_1": int, "score_2": int, "score_3": int, "reason": str}'
)


def _panel_code(panel, task):
    d = panel.deliberate(task)
    return d.code_suggestions[0].code_block if d.code_suggestions else ""


def judge3(adapter, task, ordered):
    """ordered: list of (label, code) already shuffled. Returns scores aligned to it."""
    body = f"TASK:\n{task}\n\n" + "\n\n".join(
        f"--- SOLUTION {i + 1} ---\n{code}" for i, (_lbl, code) in enumerate(ordered)
    )
    raw = adapter.generate(
        [{"role": "system", "content": _JUDGE3_SYS}, {"role": "user", "content": body}],
        max_tokens=4000, temperature=0.0,
    )
    obj = _extract_json(raw) or {}
    return [int(obj.get(f"score_{i + 1}", 0) or 0) for i in range(3)]


def run(n: int, k: int, out: Path, effort: str = "low") -> dict:
    key = NeoConfig.load().api_key
    base = ab.EffortAdapter(OpenAIAdapter(model="gpt-5.5", api_key=key), effort)
    claude = ab._critic_adapter("anthropic", "claude-sonnet-4-5-20250929")

    panel_gpt = MultiAgentReasoner(lambda role: base, k_plans=k, max_repair_rounds=1)
    panel_claude = MultiAgentReasoner(
        lambda role: (claude if role == "critic" else base), k_plans=k, max_repair_rounds=1
    )

    rng = random.Random(7)
    tasks = ab.TASKS[:n]
    rows = []
    sums = {"single": 0, "panel_gpt": 0, "panel_claude": 0}
    scored = 0

    for i, task in enumerate(tasks, 1):
        try:
            arms = [
                ("single", ab.single_call(base, task).get("code", "")),
                ("panel_gpt", _panel_code(panel_gpt, task)),
                ("panel_claude", _panel_code(panel_claude, task)),
            ]
        except Exception as e:
            print(f"[{i}/{len(tasks)}] ERROR: {type(e).__name__}: {str(e)[:120]}", flush=True)
            rows.append({"task": task, "error": f"{type(e).__name__}: {str(e)[:200]}"})
            continue

        ordered = arms[:]
        rng.shuffle(ordered)
        scores = judge3(base, task, ordered)
        by_arm = {lbl: sc for (lbl, _), sc in zip(ordered, scores)}
        for kk in sums:
            sums[kk] += by_arm[kk]
        scored += 1
        rows.append({"task": task, **by_arm})
        print(f"[{i}/{len(tasks)}] single={by_arm['single']} panel_gpt={by_arm['panel_gpt']} "
              f"panel_claude={by_arm['panel_claude']} :: {task[:42]}", flush=True)

    d = max(1, scored)
    div_c = sum(1 for r in rows if "panel_claude" in r and r["panel_claude"] > r["panel_gpt"])
    div_g = sum(1 for r in rows if "panel_claude" in r and r["panel_gpt"] > r["panel_claude"])
    summary = {
        "n": len(tasks), "scored": scored, "k": k, "effort": effort,
        "avg_single": round(sums["single"] / d, 2),
        "avg_panel_gpt": round(sums["panel_gpt"] / d, 2),
        "avg_panel_claude": round(sums["panel_claude"] / d, 2),
        "orchestration_gain_B_minus_A": round((sums["panel_gpt"] - sums["single"]) / d, 2),
        "diversity_gain_C_minus_B": round((sums["panel_claude"] - sums["panel_gpt"]) / d, 2),
        "diversity_wins_claude/gpt/tie": f"{div_c}/{div_g}/{scored - div_c - div_g}",
    }
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--out", default="/tmp/ab_controlled.json")
    a = ap.parse_args()
    run(a.n, a.k, Path(a.out), effort=a.effort)
