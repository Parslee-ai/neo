# CAR adapter, CAR model pins, provider param compatibility

Detailed engineering notes moved verbatim from `CLAUDE.md` (2026-10-12). `CLAUDE.md` keeps the short rules; this file keeps the reasons, measurements and footguns behind them.

- CarAdapter defaults `intent_hint={"task":"code","prefer_quality":True}` so CAR's router
  routes neo to the most capable model, not the chat/cost default. This is the *intended*
  router API, not a hack:
  [Parslee-ai/car-releases#52](https://github.com/Parslee-ai/car-releases/issues/52)
  (router cost-bias on `task=code`) is **closed** — the router now prioritizes
  quality > speed > cost for `task=code`, and the 0.25–0.27 reworks add capability-honest
  routing + `exclude_models`. We keep `prefer_quality` explicit as belt-and-suspenders
  (CAR's *default* profile without it is still latency/cost-biased). Rationale lives on
  `CarAdapter.DEFAULT_INTENT_HINT` in `adapters.py`. Observer floor is car-runtime ≥0.18.0,
  now enforced at runtime by `_require_car_runtime` (version check, not just the `agents_*`
  attr); latest validated against car-runtime **0.56.1** (full `test_car_adapter`
  suite including the live calls, plus every other CAR and observer suite, against a CarHost 0.56.1 daemon). Note the
  pin `car-runtime>=0.27.0,<1.0` lets the client drift from a daemon that ships
  inside CarHost.app, so a client/daemon **version skew is the normal state**,
  and it is harmless ONLY while both sides speak the same wire protocol. **It stopped being harmless at the v2→v3 protocol bump**: car-runtime 0.50.0 (v2)
  against a CarHost 0.52.1 daemon (v3) fails `server.handshake`, and `agents_list`
  then SILENTLY falls back to the manifest — every agent `stopped`, pid None — so
  `neo memory observer status` printed `stopped` for an observer nine days into
  sweeping, and stop/kick returned `not_running` without acting. CarHost updates
  itself and pipx does not, so this recurs. `observer._supervisor_blind_spot`
  cross-checks CAR's answer against `~/.neo/observer.lock` (the one liveness signal
  that does not route through CAR); on disagreement status reads `unverified`,
  start/stop/kick refuse, and a no-op `agents_stop` on a nonexistent id surfaces
  car-runtime's routing error — the only place it reports one. A hand-rolled
  WebSocket client that skips the handshake was rejected: it works today only
  because the daemon tolerates the missing handshake. Updating the
  daemon means updating CarHost.app — there is no `car` CLI to run and nothing
  neo can do about it. Do not paper over it with `CAR_NO_VERSION_WARNING=1`; the
  warning is accurate.
- **CAR model pins** (`CarAdapter`, #257/#261/#267): CAR has two pin fields.
  `model_id` is an exact catalog-id pin — CAR serves only that row and echoes it
  as `requested_model_id`; `model` is the display-name/alias path, where a short
  name can resolve to another row and `requested_model_id` is null by design.
  **Exactness is decided by PROVENANCE, never by the id's shape**: an id CAR's
  router handed us is passed as `CarAdapter(model_is_catalog_id=True)` (the
  panel's role factory does this for every router-plan id) and goes out as
  top-level `model_id` through `infer_tracked_with_request` — car-runtime 0.55's
  `infer_tracked` has no `model_id` keyword — and is accepted only when
  `requested_model_id == resolved_model_id == pin`, by equality with no parsing.
  CAR guarantees `provider/name[:tag]` only for BUILT-IN ids; discovered,
  gateway and operator ids pass through verbatim (`parslee/openrouter/open-fast`,
  `:` inside a bare id, possibly no `/`), which is why the first cut's
  "contains `/`" test was replaced. A name typed into config stays on `model`,
  with an equality-first check and then an exact NAME match against
  `resolved_model_id` (`_resolved_id_honors_pin`: provider/tag only when the pin
  states them). **`model_used` is a DISPLAY name and must never be the
  evidence when an id is available**: the router's own choice
  `mlx/qwen3-0.6b:6bit` is served as `Qwen3-0.6B-MLX`, the old fuzzy check
  refused it, and every panel that drew it fell back to the fast path. The fuzzy
  `model_used` rule survives only for CAR builds that report no
  `resolved_model_id`. No echo at all on a `model_id` request means a daemon
  that predates exact pinning, and fails closed with an "update CarHost" error
  rather than blaming the pin. **No lenient fallback** (Matt's decision): the
  `resolved_model_id` format is a CAR-side contract, and tolerating "close
  enough" ids is how a `gpt-5` pin passed when served `gpt-5.5`. Sending `model`
  and `model_id` together is rejected by CAR; an unknown either fails `model not
  found` (CAR sets `strict_model`), never a silent default. Any new caller that
  holds a router-provided id must pass `model_is_catalog_id=True`.
- Provider param compatibility (`adapters.py`): providers reject optional request
  fields model by model — `temperature` (Anthropic Opus 4.7+/Sonnet 5/Fable 5;
  OpenAI o-series/gpt-5/gpt-6; Azure reasoning deployments; xAI Grok/DeepSeek
  reasoners), `reasoning.effort` (every non-reasoning model), `max_tokens` (chat
  completions wants `max_completion_tokens`). There's no reliable model-string
  rule (opus-4-6 accepts `temperature`, opus-4-7 rejects it; Azure `model` is an
  arbitrary deployment name), so **no adapter branches on a model name, and there
  is no allow-list of reasoning models** — that is how `gpt-6.1-sol` silently lost
  its effort: the endpoint and fields were picked by `"gpt-5" in model`, and the
  new name missed the test. Instead ONE helper, `_create_resilient(create, kwargs,
  provider, model, optional=, renames=, ladders=)`, serves OpenAI (responses),
  Azure, Local, Anthropic and Google: it sends everything, and when a rejection
  NAMES a field Neo sent it drops, renames or lowers that one field, retries, and
  persists the learning in `_ModelParamCompat` (`~/.neo/model_param_compat.json`,
  `"<provider>:<model>" → [flags]`) so the retry is paid once per model. Flags are
  `drop_<field>` / `rename_<field>` (optionally `@<ladder field>=<level>`: a drop
  learned while effort was sent holds only at that level) and
  `refuse_<field>:<level>` (that one level is refused; step down past it, levels
  are never assumed contiguous). Stored refusals are applied to a fixed point
  BEFORE drops, so a conditional drop is judged against the level actually sent.
  Older `max_` caps are ignored; a 400 whose code is one of OpenAI's value-range
  codes (`_VALUE_ERROR_CODES`) is a bad value, never a drop — a denylist, because
  vLLM/xAI behind LocalAdapter send their own codes. **Nothing is persisted
  until the adapted call has SUCCEEDED**: learnings are collected during the retry
  walk and written once at the end, because they used to be written before the
  retry proved them, so a call that then failed (a context-length 400 whose text
  mentioned temperature) left a false `drop_temperature` on disk. **When the 400
  carries a structured `body.param`, only the field equal to it can match and the
  message cannot match a different field** (message text is the fallback only
  when there is no `param`: Anthropic, Google, client-side TypeError; for a
  rename field it is still read after `param` matches, where it can only veto);
  a message that merely mentions a field is not a rejection of it. Never a blanket
  strip-and-retry, and anything that is not a rejection of a sent field (auth,
  429, another 400, an unrelated TypeError) re-raises untouched. Bounded by
  construction: each recovery removes a field, renames it away or lowers a level,
  and one that changes nothing ends the loop. Three rejection shapes, all handled
  there: an HTTP 400 naming the field (read `body.param` as well as the message —
  `Invalid value: 'ultra'. Supported values are…` never names `reasoning.effort`
  in its text); the SDK refusing the keyword at the signature (a client-side
  `TypeError`, which is what `anthropic` 1.x does to `temperature`, so a
  `BadRequestError`-only handler never sees it); and `code`, which separates "this
  LEVEL is not offered" (`unsupported_value` is what a genuinely unsupported level
  returns live; `invalid_value` is what a value outside the known vocabulary
  returns; both → step down `EFFORT_LEVELS` one rung, persisted as
  `refuse_reasoning.effort:<level>` once the call succeeds: the REFUSED level, not a
  cap, so levels are never assumed contiguous (#266); legacy `max_` caps from
  older versions are still applied) from "this field is unsupported"
  (`unsupported_parameter` or None — `temperature`'s code is None, so `code` alone
  can never recognise an unsupported field). Any other `code` (`decimal_above_max_value`
  …) is a VALUE error on a supported field and is never learned as a drop (#266).
  **A rejected level is never evidence that effort is unsupported.** Live,
  `gpt-6.1-sol` refuses level `none` with `unsupported_value` while accepting
  `xhigh`; the walk used to persist `drop_reasoning.effort` at the bottom, after
  which every later `xhigh` request silently ran at the provider default. A
  refusal at the bottom rung is now `refuse_reasoning.effort:none`, which removes
  effort only for a request AT that level. A model whose FLOOR is above the
  request (`gpt-5.2-pro`/`gpt-5.4-pro`/`gpt-5.5-pro` accept `medium`..`xhigh`,
  `gpt-5-pro` only `high`) therefore pays the walk once and then runs at the
  provider default in one request (#265). A drop learned while an effort level was
  sent is conditional on it (`drop_temperature@reasoning.effort=high`), because
  `gpt-5.1`/`gpt-5.2` accept `temperature` only at effort `none` (#266). Stored
  flags apply levels first to a fixed point, then renames and drops, and flags
  naming a field the adapter does not adapt are ignored. `stop_sequences` is sent but not droppable: no provider is known to reject
  it. **`provider=openai` with a custom `base_url` now requires a server that
  implements the responses endpoint**; `provider=local` is the route for
  OpenAI-compatible servers that speak only chat completions.
  **`OpenAIAdapter` never calls chat completions**: every model goes
  through `client.responses.create`, which is the only endpoint carrying effort;
  `stop` is accepted and not sent (responses has no stop sequences). Azure and
  Local deliberately stay on chat completions (deployment API versions lag; most
  OpenAI-compatible servers speak only chat). Exempt: Ollama (raw HTTP with
  `options`, which ignores unknown ones), ClaudeCode (shells out to the CLI), CAR
  and Auto (Auto delegates). CAR is exempt from the RETRY helper, not from the
  fields: `CarAdapter` sends `params.temperature` through
  `infer_tracked_with_request` (the keyword `infer_tracked` has no sampling
  knobs), and CAR drops it per backend itself. Measured on 0.56.1 rather than
  assumed from "no error", since CAR silently ignores unknown fields: temperature
  IS forwarded (raw-runtime probes: local and hosted rows deterministic at 0.0,
  varied at 1.5+; through `CarAdapter` itself the recorded request carries it and
  output matches the raw request 6/6);
  `stop` is accepted and NOT applied; there is no effort level and `thinking`
  has no observable effect. So neither of the last two is sent
  (car-releases#107/#108).
  **Footguns**: the 400 test requires `status_code == 400` (google-genai reports
  it as int `code`), so a fake exception in a test must carry one; and the store
  is best-effort by design (I/O failure → in-memory only, never breaks inference),
  merge-on-write + atomic `os.replace`, path resolved at call time. A provider
  returning 422 for a param error won't be caught.
