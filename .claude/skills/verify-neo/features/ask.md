# Ask Neo (reasoning request)

A plain `neo "<question>"` gathers context, calls the configured provider, and returns an answer. With `--json` the answer is one stdout document and progress is JSONL on stderr.

## Sub-features

- `ask-nokey` fails clearly when no provider is configured.
- `ask-fast` answers through the single-call path (`--fast`).
- `ask-json` returns one stdout JSON document.

## How to get to it (user POV)

- Run `neo "<question>"` in a repo.
- Hosts run `neo --json --fast "<question>"`.

## Driving it with verify.sh

Preconditions:

- Baseline state from the README.
- For `ask-fast` and `ask-json`: a provider key exported in the calling shell (for example `OPENAI_API_KEY`, or `NEO_PROVIDER` + `NEO_MODEL` + that provider's key). The helper passes it through. Each call costs money.

- **No key.** With no provider key exported, run `$V neo "$ID" ask-nokey --fast --json "What does invoice_total return when days_late is 0?"`. Prints `exit=1`; stdout is a JSON object whose `error` reads `Failed to initialize LM adapter: OpenAI API key required` and has a `hint`.
- **Real answer.** With a key exported, run `$V neo "$ID" ask --fast --json "What does invoice_total return when days_late is 0?"`. Prints `exit=0`; stdout parses as one JSON document; stderr ends with a `completed` event. The answer says it returns `subtotal` unchanged.
- **Learning side effect.** Run `$V snapshot "$ID" after-ask`. Because the default mode is `learn`, files appear under `home/.neo/` (episodes or sessions). Record which.

## Gotchas

- The scratch HOME has no `config.json`, so the default provider is OpenAI. A key in `~/.neo/config.json` is NOT visible to the run.
- Without `--fast`, a reachable `car-server` can make Neo run the multi-agent panel: slower and more expensive.
- Never put a real key into a label or prompt; `.cmd` files are evidence and are kept.
