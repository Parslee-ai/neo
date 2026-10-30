# CAR-first gate re-run — CAR 0.58.0 (2026-10-24)

**Verdict: parity NOT met. `inference_mode` stays `static`.** Same harness,
setup and judge as `car-router-eval-2026-10-07.md` (`tools/ab_car_router.py`,
six real neo prompts captured fresh against today's repo).

| | CAR router | static gpt-6.1-sol |
|---|---|---|
| Model served | `claude-opus-4-8` 5/6, `claude-opus-4-7` 1/6 | gpt-6.1-sol |
| Errors / small-model routes | 0 / 0 | 0 |
| Schema parses | 6/6 | 6/6 |
| Head-to-head wins | **0** | **6** |
| Mean judge score | 3.0 | 8.83 |

Unchanged from 0.56.1. Five of six judge reasons repeat the earlier failure:
the routed answer asserts root causes or writes diffs against truncated code it
did not see, where the static answer marks the gap and plans an inspection. The
one close prompt (an explanation, no code to invent) scored 7 vs 9. One prompt
fell back to opus-4-7 and took 372 s.

Upstream issues car-releases#105 (code quality scores), #106 (no GPT-6 rows)
and #107 (no reasoning-effort param) are all still open. Re-run on the next CAR
release that closes one of them.
