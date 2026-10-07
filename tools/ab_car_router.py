#!/usr/bin/env python3
"""CAR-first gate: does CAR's router match neo's static model on REAL neo prompts?

neo's ``inference_mode`` defaults to ``static`` (``NeoConfig().model`` through
``OpenAIAdapter``). ``auto`` routes through CAR first, and stays opt-in until a
CAR release is shown not to lose quality: on car-runtime 0.23.0 the router
cost-biased ``task=code`` onto mini models and lost ~70% of head-to-heads. That
first eval was run ad hoc and never committed, so it could not be re-run; this
harness is the reproducible replacement.

Both arms are what PRODUCTION builds, not an equalized comparison -- the
question is "would flipping the default hurt", not "are two models equal under
identical knobs":

    A  static = OpenAIAdapter(NeoConfig.load().model), the engine's recorded
                max_tokens / temperature / reasoning_effort
    B  car    = CarAdapter(model=None) -- exactly what ``resolve_adapter``
                builds for ``auto`` (default intent ``task=code`` +
                ``prefer_quality``; CAR ignores the sampling knobs, in
                production too). Called DIRECTLY, never through AutoAdapter:
                a CAR failure there falls back to the static model and arm B
                silently becomes arm A.

Prompts are REAL: each is captured with ``neo --dry-run --json`` against a
repository, so the messages are byte-for-byte what the engine hands the
adapter (repository context included -- ~130K chars on this repo). Routing
depends on prompt size, so a toy prompt would measure a different router
decision than the one production gets.

Three measurements per prompt:
  1. ROUTING  -- which model CAR served (``resolved_model_id`` / ``model_used``),
     latency, tokens. ``--probe`` stops here: the 0.23.0 defect was a routing
     defect, and this is the cheap decisive check.
  2. SCHEMA   -- whether the output parses into neo's three ``<<<NEO:SCHEMA``
     blocks with neo's own parser. Objective and judge-free; it is where weaker
     models fail.
  3. JUDGE    -- blind pairwise score, order randomized per prompt (seeded).
     The default judge is the static model judging its own arm, so
     self-preference favours A: a CAR tie or win under it is robust, a CAR loss
     is partly confounded. ``--judge-provider anthropic`` uses a third party.

PARITY (decided before running): CAR does not lose a majority of decided
head-to-heads, its schema compliance is not lower than static's, and no prompt
was routed to a mini/small model. n is small (6 by default); report
win/lose/tie and do not overclaim.

Usage:
    python tools/ab_car_router.py --probe
    python tools/ab_car_router.py --out results.json
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from neo.adapters import CarAdapter, OpenAIAdapter  # noqa: E402
from neo.config import NeoConfig  # noqa: E402
from neo.engine import NeoEngine  # noqa: E402
from neo.multi_agent import _extract_json  # noqa: E402
from neo.structured_parser import (  # noqa: E402
    parse_code_suggestions,
    parse_plan_steps,
    parse_simulation_traces,
)

# The kinds of request neo actually gets: bugfix, explanation, refactor,
# feature, diagnosis, optimization -- each naming real code in this repo.
PROMPTS = [
    "fix the bug where _create_resilient persists a learned flag before the retried call succeeds",
    "explain how FactStore decides when an episode candidate promotes to a durable pattern",
    "refactor collect_outcomes so the host edit ledger read is a separate helper",
    "add a --limit flag to neo memory citation-stats that caps the number of rows printed",
    "why does the observer re-exec itself every N cycles and what breaks if the lock is held across exec",
    "optimize _history_boost in context_gatherer, it is slow on large repos",
]

# Model-name fragments that mark the cost-biased routing the gate exists for.
SMALL_MODEL_MARKERS = ("mini", "nano", "small", "haiku", "flash-lite", "0.6b", "1b", "3b", "7b", "8b")

_JUDGE_SYS = (
    "You are a strict senior reviewer. Two assistants answered the same request about "
    "a real repository; the repository context they were given is included. Judge each "
    "answer 1-10 on correctness against that code first, then whether its plan and "
    "code changes would actually solve the request, then completeness. Penalize claims "
    "the context contradicts and invented APIs heavily. Ignore length and formatting. "
    'Respond ONLY with JSON: {"score_a": int, "score_b": int, '
    '"winner": "A"|"B"|"tie", "reason": str}'
)

CALL_TIMEOUT_S = 600


class RecordingRuntime:
    """Forward to CAR's real runtime and keep each parsed result, so the
    harness sees which model served the call through the real CarAdapter."""

    def __init__(self, inner):
        self._inner = inner
        self.last: dict = {}

    def _keep(self, raw):
        try:
            self.last = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except (ValueError, TypeError):
            self.last = {}
        return raw

    def infer_tracked(self, *a, **kw):
        return self._keep(self._inner.infer_tracked(*a, **kw))

    def infer_tracked_with_request(self, *a, **kw):
        return self._keep(self._inner.infer_tracked_with_request(*a, **kw))


def capture_prompts(cwd: Path, cache: Path) -> list[dict]:
    """Capture each prompt's exact adapter input with ``neo --dry-run --json``."""
    cache.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "NEO_OBSERVER_AUTOSTART": "0", "NEO_SKIP_UPDATE_CHECK": "1"}
    out = []
    for i, prompt in enumerate(PROMPTS, 1):
        path = cache / f"p{i}.json"
        if not path.exists():
            r = subprocess.run(
                [sys.executable, "-m", "neo", "--dry-run", "--json", prompt, "--cwd", str(cwd)],
                capture_output=True, text=True, timeout=600, env=env,
            )
            if r.returncode != 0:
                raise RuntimeError(f"dry-run failed for prompt {i}: {r.stderr[-500:]}")
            path.write_text(r.stdout)
        call = json.loads(path.read_text())["calls"][0]
        out.append({"prompt": prompt, "call": call})
    return out


def schema_ok(text: str) -> dict:
    """Parse exactly as the engine's fast path does: split into sections with
    ``NeoEngine._extract_section`` first, then parse each alone. Handing the
    whole response to a block parser fails every block after the first as
    "text before sentinel" -- a harness artifact, not a model failure."""
    split = NeoEngine._extract_section  # stateless; reused, never restated
    parsed = {
        "plan": parse_plan_steps(split(None, text, "plan")).success,
        "simulation": parse_simulation_traces(split(None, text, "simulation")).success,
        "code": parse_code_suggestions(split(None, text, "code")).success,
    }
    parsed["all"] = all(parsed.values())
    return parsed


def timed(fn):
    with cf.ThreadPoolExecutor(max_workers=1) as ex:
        t0 = time.time()
        fut = ex.submit(fn)
        try:
            return fut.result(timeout=CALL_TIMEOUT_S), time.time() - t0, None
        except Exception as e:  # noqa: BLE001 - record and keep going
            return "", time.time() - t0, f"{type(e).__name__}: {str(e)[:300]}"


def call_kwargs(call: dict) -> dict:
    return {
        "max_tokens": call.get("max_tokens") or 8192,
        "temperature": call.get("temperature", 0.3),
        "reasoning_effort": call.get("reasoning_effort"),
        "stop": call.get("stop"),
    }


def judge_adapter(provider: str):
    if provider == "anthropic":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import ab_reasoning as ab

        return ab._critic_adapter("anthropic", "")
    cfg = NeoConfig.load()
    return OpenAIAdapter(model=cfg.model, api_key=cfg.api_key)


def judge(adapter, messages, sol_a: str, sol_b: str) -> dict:
    request = "\n\n".join(m["content"] for m in messages if m.get("role") == "user")
    user = (f"REQUEST AND REPOSITORY CONTEXT:\n{request}\n\n"
            f"--- ANSWER A ---\n{sol_a}\n\n--- ANSWER B ---\n{sol_b}")
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


def routed_model(rec: dict) -> str:
    return rec.get("resolved_model_id") or rec.get("model_used") or "unknown"


def is_small(model: str) -> bool:
    low = model.lower()
    return any(m in low for m in SMALL_MODEL_MARKERS)


def run(args) -> dict:
    from neo.car_inference import get_runtime

    cases = capture_prompts(Path(args.cwd).resolve(), Path(args.cache))
    rec_rt = RecordingRuntime(get_runtime())
    car = CarAdapter(model=None, runtime=rec_rt)
    cfg = NeoConfig.load()
    static = OpenAIAdapter(model=cfg.model, api_key=cfg.api_key)
    judge_lm = None if args.probe else judge_adapter(args.judge_provider)
    rng = random.Random(1234)

    rows = []
    for i, case in enumerate(cases, 1):
        messages, kw = case["call"]["messages"], call_kwargs(case["call"])
        rec_rt.last = {}
        car_text, car_s, car_err = timed(lambda: car.generate(messages, **kw))
        info = rec_rt.last
        row = {
            "prompt": case["prompt"],
            "prompt_chars": sum(len(m["content"]) for m in messages),
            "car": {
                "model": routed_model(info), "model_used": info.get("model_used"),
                "resolved_model_id": info.get("resolved_model_id"),
                "seconds": round(car_s, 1), "usage": info.get("usage"),
                "error": car_err, "schema": schema_ok(car_text) if car_text else None,
                "text": car_text,
            },
        }
        print(f"[{i}/{len(cases)}] car -> {row['car']['model']} in {car_s:.0f}s"
              f"{' ERROR ' + car_err if car_err else ''} schema={row['car']['schema']}", flush=True)
        if not args.probe:
            st_text, st_s, st_err = timed(lambda: static.generate(messages, **kw))
            row["static"] = {"model": cfg.model, "seconds": round(st_s, 1), "error": st_err,
                             "schema": schema_ok(st_text) if st_text else None,
                             "text": st_text}
            print(f"        static -> {cfg.model} in {st_s:.0f}s"
                  f"{' ERROR ' + st_err if st_err else ''} schema={row['static']['schema']}",
                  flush=True)
            if car_text and st_text:
                car_is_a = rng.random() < 0.5
                a, b = (car_text, st_text) if car_is_a else (st_text, car_text)
                v, _, j_err = timed(lambda: judge(judge_lm, messages, a, b))
                if j_err:
                    row["judge_error"] = j_err
                else:
                    car_won = None if v["winner"] == "tie" else (v["winner"] == "A") == car_is_a
                    row["judge"] = {
                        "car_is_a": car_is_a, "winner": v["winner"], "car_won": car_won,
                        "car_score": v["score_a"] if car_is_a else v["score_b"],
                        "static_score": v["score_b"] if car_is_a else v["score_a"],
                        "reason": v["reason"],
                    }
                    print(f"        judge: car {row['judge']['car_score']} vs static "
                          f"{row['judge']['static_score']} ({v['winner']}, car_won={car_won})",
                          flush=True)
        rows.append(row)

    summary = summarize(rows, args.probe)
    report = {"static_model": cfg.model, "judge_provider": None if args.probe else args.judge_provider,
              "summary": summary, "rows": rows}
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2))
    return report


def summarize(rows: list[dict], probe: bool) -> dict:
    routed = [r["car"]["model"] for r in rows if not r["car"]["error"]]
    s = {
        "n": len(rows),
        "car_errors": sum(1 for r in rows if r["car"]["error"]),
        "car_models": sorted(set(routed)),
        "small_model_routes": sum(1 for m in routed if is_small(m)),
        "car_schema_ok": sum(1 for r in rows if (r["car"]["schema"] or {}).get("all")),
    }
    if probe:
        return s
    judged = [r["judge"] for r in rows if "judge" in r]
    s.update({
        "static_errors": sum(1 for r in rows if r["static"]["error"]),
        "static_schema_ok": sum(1 for r in rows if (r["static"]["schema"] or {}).get("all")),
        "judged": len(judged),
        "car_wins": sum(1 for j in judged if j["car_won"] is True),
        "static_wins": sum(1 for j in judged if j["car_won"] is False),
        "ties": sum(1 for j in judged if j["car_won"] is None),
        "car_mean_score": round(sum(j["car_score"] for j in judged) / len(judged), 2) if judged else None,
        "static_mean_score": round(sum(j["static_score"] for j in judged) / len(judged), 2) if judged else None,
    })
    s["parity"] = (
        s["car_errors"] == 0 and s["small_model_routes"] == 0
        and s["car_schema_ok"] >= s["static_schema_ok"]
        and s["car_wins"] + s["ties"] >= s["static_wins"]
    )
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--probe", action="store_true", help="CAR routing + schema only; no static arm, no judge")
    ap.add_argument("--cwd", default=str(REPO), help="repository the prompts are captured against")
    ap.add_argument("--cache", default=str(REPO / ".neo" / "ab_car_router"),
                    help="where captured dry-run prompts are kept")
    ap.add_argument("--judge-provider", choices=["openai", "anthropic"], default="openai")
    ap.add_argument("--out", default="")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
