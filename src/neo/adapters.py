"""
LM Adapter implementations for OpenAI, Anthropic, and local models.
"""

import os
import json
import logging
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

# Load environment variables from .env file
try:
    from neo.load_env import load_env
    load_env()
except ImportError:
    pass

from neo.models import LMAdapter

logger = logging.getLogger(__name__)


# ============================================================================
# Reasoning-model parameter compatibility (OpenAI-SDK adapters)
# ============================================================================
#
# Newer reasoning models reject standard chat-completions parameters:
#   - `temperature` is rejected (OpenAI o-series / gpt-5, xAI Grok and DeepSeek
#     reasoners behind an OpenAI-compatible endpoint, Azure reasoning deploys,
#     Anthropic Opus 4.7+ / Sonnet 5 / Fable 5).
#   - `max_tokens` must be sent as `max_completion_tokens` (OpenAI-family).
# There is no reliable model-string rule — and on Azure `model` is an arbitrary
# deployment name — so we learn each adaptation from the API's own 400. The
# learnings are persisted (see `_ModelParamCompat`) so the first-call retry
# penalty isn't re-paid on every short-lived CLI invocation.

# Adaptation flags recorded per model.
_ADAPT_DROP_TEMPERATURE = "drop_temperature"
_ADAPT_RENAME_MAX_TOKENS = "rename_max_tokens"


class _ModelParamCompat:
    """Persistent record of which parameter adaptations each model needs.

    Backed by ``~/.neo/model_param_compat.json`` as ``{"<provider>:<model>":
    ["<flag>", ...]}``. Keyed by provider so a bare model name (e.g. ``gpt-4``)
    behind a Local endpoint can't collide with the same name on Azure/OpenAI.

    - The path is resolved at call time so per-test ``Path.home()`` stubs apply
      (mirrors ``neo.memory.metrics._metrics_path``).
    - An in-memory cache avoids re-reading on every call; it reloads whenever
      the resolved path changes (so each isolated test home starts clean, while
      production loads once).
    - Writes are merge-on-write + atomic replace, so concurrent neo processes
      union their learnings rather than clobbering each other.
    - Persistence is best-effort: any I/O failure degrades to in-memory-only
      and never breaks inference.
    """

    def __init__(self) -> None:
        self._path: Optional[Path] = None
        self._data: dict[str, set[str]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _resolve() -> Path:
        return Path.home() / ".neo" / "model_param_compat.json"

    @staticmethod
    def _read(path: Path) -> dict[str, set[str]]:
        """Load the store, tolerating anything. A missing file, invalid JSON,
        or valid JSON of the wrong shape (``null``, a list, a scalar, non-list
        values, unhashable flags) all degrade to an empty mapping — never a
        raise. This cache must never break inference (see `has`/`learn`)."""
        try:
            raw = json.loads(path.read_text())
        except (OSError, ValueError):
            return {}
        if not isinstance(raw, dict):
            return {}
        out: dict[str, set[str]] = {}
        for key, flags in raw.items():
            if isinstance(key, str) and isinstance(flags, list):
                out[key] = {f for f in flags if isinstance(f, str)}
        return out

    def _ensure_loaded(self, path: Path) -> None:
        if path != self._path:
            self._data = self._read(path)
            self._path = path

    def has(self, provider: str, model: str, adaptation: str) -> bool:
        # Totally guarded: a compat cache must never propagate a failure into
        # the inference path. Any error (unreadable ~/.neo, no resolvable home,
        # …) degrades to "not learned" so the caller sends the param as usual.
        try:
            path = self._resolve()
            with self._lock:
                self._ensure_loaded(path)
                return adaptation in self._data.get(f"{provider}:{model}", ())
        except Exception:
            logger.debug("param-compat has() failed; assuming not-learned", exc_info=True)
            return False

    def learn(self, provider: str, model: str, adaptation: str) -> None:
        try:
            path = self._resolve()
            key = f"{provider}:{model}"
            with self._lock:
                self._ensure_loaded(path)
                if adaptation in self._data.get(key, ()):
                    return
                self._data.setdefault(key, set()).add(adaptation)
                self._persist(path)
        except Exception:
            logger.debug("param-compat learn() failed; skipping", exc_info=True)

    def _persist(self, path: Path) -> None:
        # Merge with whatever's on disk now — another process may have learned
        # something since we loaded — then atomically replace. Deliberately no
        # flock (unlike FactStore): writes are rare and additive-only, so the
        # merge self-heals a lost update (the loser re-persists its own
        # `self._data` on its next learn()). Not worth locking the hot path for.
        merged = self._read(path)
        for key, flags in self._data.items():
            merged.setdefault(key, set()).update(flags)
        self._data = merged
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump({k: sorted(v) for k, v in merged.items()}, f)
                os.replace(tmp, str(path))
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except OSError:
            pass  # best-effort; in-memory cache still holds the learning


_PARAM_COMPAT = _ModelParamCompat()


def _chat_completion_resilient(client, kwargs: dict, provider: str):
    """Call ``client.chat.completions.create(**kwargs)``, recovering from the
    reasoning-model parameter rejections described above.

    Adaptations already learned for ``provider``/model are applied up front;
    anything new is discovered from the 400, applied, remembered, and retried.
    A 400 for any other reason re-raises untouched. Loop is bounded implicitly:
    each adaptation removes the key it keys on, so it can fire at most once
    (worst case: temperature + max_tokens + success = 3 calls). ``kwargs`` is
    copied, not mutated, so callers may safely pass a shared/cached dict.

    Scope of each adaptation:
      - `temperature` drop is broad — o-series / gpt-5, Azure reasoning
        deployments, and OpenAI-compatible reasoners (xAI Grok, DeepSeek) all
        reject it.
      - `max_completion_tokens` rename is OpenAI-family only (OpenAI + Azure,
        which share the `openai` SDK's error text). Grok/DeepSeek accept
        `max_tokens`, so they never hit that branch.
    """
    import openai

    kwargs = dict(kwargs)  # own our copy — never mutate the caller's dict
    model = kwargs.get("model", "")
    if _PARAM_COMPAT.has(provider, model, _ADAPT_DROP_TEMPERATURE):
        kwargs.pop("temperature", None)
    if _PARAM_COMPAT.has(provider, model, _ADAPT_RENAME_MAX_TOKENS) and "max_tokens" in kwargs:
        kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")

    while True:
        try:
            return client.chat.completions.create(**kwargs)
        except openai.BadRequestError as e:
            msg = str(e).lower()
            # NOTE: we can't distinguish "temperature is deprecated/unsupported"
            # from an out-of-range value error ("temperature must be <= 2") by
            # message alone. That's fine here: neo only ever sends in-range
            # temperatures, so a `temperature` 400 always means rejection.
            if "temperature" in kwargs and "temperature" in msg:
                _PARAM_COMPAT.learn(provider, model, _ADAPT_DROP_TEMPERATURE)
                kwargs.pop("temperature", None)
                logger.debug(
                    "Model %s/%s rejected `temperature`; dropped it and retrying",
                    provider, model,
                )
                continue
            # Rename trigger: the message references `max_tokens` and signals
            # the param is unsupported (either by naming the replacement or by
            # an unsupported/not-supported phrase — robust to minor OpenAI
            # wording changes). A plain value error on max_tokens carries none
            # of these and correctly re-raises.
            if "max_tokens" in kwargs and "max_tokens" in msg and (
                "max_completion_tokens" in msg
                or "unsupported" in msg
                or "not supported" in msg
            ):
                _PARAM_COMPAT.learn(provider, model, _ADAPT_RENAME_MAX_TOKENS)
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
                logger.debug(
                    "Model %s/%s requires `max_completion_tokens`; renamed and "
                    "retrying", provider, model,
                )
                continue
            raise


# ============================================================================
# OpenAI Adapter
# ============================================================================

class OpenAIAdapter(LMAdapter):
    """Adapter for OpenAI models (GPT-4, GPT-5, etc.)."""

    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url

        if not self.api_key:
            raise ValueError("OpenAI API key required")

        try:
            import openai
            kwargs = {"api_key": self.api_key}
            if base_url:
                kwargs["base_url"] = base_url
            self.client = openai.OpenAI(**kwargs)
        except ImportError:
            raise ImportError("openai package required: pip install openai")

    def generate(
        self,
        messages: list[dict[str, str]],
        stop: Optional[list[str]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        """Generate response using OpenAI API."""
        from neo.memory.metrics import capture_lm_call_failure
        with capture_lm_call_failure(
            provider="openai",
            model=self.model,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        ):
            # gpt-5* and codex models use /v1/responses endpoint
            if "codex" in self.model.lower() or "gpt-5" in self.model.lower():
                import httpx
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                base_url = self.base_url or "https://api.openai.com"
                url = f"{base_url}/v1/responses"

                payload: dict = {
                    "model": self.model,
                    "input": messages,
                    "max_output_tokens": max_tokens,
                }
                # gpt-5* and codex models reject `temperature` on /v1/responses
                # with a 400 ("Unsupported parameter"). Their reasoning behavior
                # is steered by `reasoning.effort` instead. Don't include it.
                if reasoning_effort is not None:
                    payload["reasoning"] = {"effort": reasoning_effort}

                response = httpx.post(url, headers=headers, json=payload, timeout=600.0)  # 10 minutes for complex queries
                if response.status_code != 200:
                    raise ValueError(f"API error {response.status_code}: {response.text}")
                data = response.json()
                self._emit_usage_metric(data.get("usage", {}))

                # Extract text from output array
                output = data.get("output", [])
                for item in output:
                    if item.get("type") == "message" and item.get("status") == "completed":
                        content = item.get("content", [])
                        for c in content:
                            if c.get("type") == "output_text":
                                return c.get("text", "")

                raise ValueError(f"No completed message in response: {data}")
            else:
                # Standard chat completions for other models. Route through the
                # resilient helper so o-series (and other reasoning models that
                # aren't matched above) recover from `temperature` /
                # `max_tokens` rejections instead of erroring.
                response = _chat_completion_resilient(self.client, {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stop": stop,
                }, provider="openai")
                self._emit_usage_metric(getattr(response, "usage", None))
                return response.choices[0].message.content

    def _emit_usage_metric(self, usage: object) -> None:
        """Record per-call token usage to metrics.jsonl.

        Handles both response shapes the OpenAI client surfaces:
          - /v1/chat/completions: response.usage.prompt_tokens,
            completion_tokens, prompt_tokens_details.cached_tokens
          - /v1/responses (gpt-5*/codex): usage dict with input_tokens,
            output_tokens, input_tokens_details.cached_tokens

        cache_hit_rate = cached / (prompt_or_input + cached) — same
        normalization shape as AnthropicAdapter so the metric is
        directly comparable across providers.
        """
        try:
            from neo.memory.metrics import record as metrics_record

            if usage is None:
                return

            # Best-effort field extraction — supports both dict and object
            # shapes (httpx returns dict, openai SDK returns Pydantic).
            def get(obj: object, *path: str) -> object:
                cur = obj
                for key in path:
                    if cur is None:
                        return None
                    if isinstance(cur, dict):
                        cur = cur.get(key)
                    else:
                        cur = getattr(cur, key, None)
                return cur

            input_tokens = (
                get(usage, "prompt_tokens")
                or get(usage, "input_tokens")
                or 0
            )
            output_tokens = (
                get(usage, "completion_tokens")
                or get(usage, "output_tokens")
                or 0
            )
            cached = (
                get(usage, "prompt_tokens_details", "cached_tokens")
                or get(usage, "input_tokens_details", "cached_tokens")
                or 0
            )
            input_tokens = int(input_tokens or 0)
            output_tokens = int(output_tokens or 0)
            cached = int(cached or 0)
            cache_hit_rate = (
                cached / input_tokens if input_tokens > 0 else 0.0
            )
            metrics_record(
                "lm_call",
                provider="openai",
                model=self.model,
                status="success",
                input_tokens=input_tokens,
                cache_read_input_tokens=cached,
                output_tokens=output_tokens,
                cache_hit_rate=round(cache_hit_rate, 4),
            )
        except Exception:
            pass  # Metrics are never load-bearing.

    def name(self) -> str:
        return f"openai/{self.model}"


# ============================================================================
# Anthropic Adapter
# ============================================================================

class AnthropicAdapter(LMAdapter):
    """Adapter for Anthropic models (Claude).

    Newer Claude models (Opus 4.7+, Sonnet 5, Fable 5) reject the `temperature`
    sampling parameter with a 400. There's no clean model-string rule (opus-4-6
    accepts it, opus-4-7 rejects it), so rather than hardcode a taxonomy that
    goes stale on the next release, we learn it from the API's own error and
    remember it via the persistent `_PARAM_COMPAT` store.
    """

    def __init__(self, model: str = "claude-sonnet-4-5-20250929", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key required")

        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")

    def generate(
        self,
        messages: list[dict[str, str]],
        stop: Optional[list[str]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: Optional[str] = None,  # not supported; accepted for ABC compat
    ) -> str:
        """Generate response using Anthropic API."""
        # Convert messages format if needed
        system_message = None
        formatted_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                formatted_messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })

        kwargs = {
            "model": self.model,
            "messages": formatted_messages,
            "max_tokens": max_tokens,
        }
        # Only send `temperature` to models not already known to reject it
        # (Opus 4.7+, Sonnet 5, Fable 5). The learn-and-retry below records
        # rejections into the persistent `_PARAM_COMPAT` store.
        if not _PARAM_COMPAT.has("anthropic", self.model, _ADAPT_DROP_TEMPERATURE):
            kwargs["temperature"] = temperature

        if system_message:
            kwargs["system"] = system_message

        if stop:
            kwargs["stop_sequences"] = stop

        import anthropic
        from neo.memory.metrics import capture_lm_call_failure
        with capture_lm_call_failure(
            provider="anthropic",
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
        ):
            try:
                response = self.client.messages.create(**kwargs)
            except anthropic.BadRequestError as e:
                # Newer Claude models reject `temperature` with a 400. Learn
                # the model, drop the param, and retry once so the call still
                # succeeds. A 400 for any other reason re-raises untouched.
                if "temperature" not in kwargs or "temperature" not in str(e).lower():
                    raise
                _PARAM_COMPAT.learn("anthropic", self.model, _ADAPT_DROP_TEMPERATURE)
                kwargs.pop("temperature", None)
                logger.debug(
                    "Anthropic model %s rejected `temperature`; dropped it "
                    "and retrying", self.model,
                )
                response = self.client.messages.create(**kwargs)
            self._emit_usage_metric(response)
            return response.content[0].text

    def _emit_usage_metric(self, response: object) -> None:
        """Record per-call token usage to metrics.jsonl.

        Captures cache-read / cache-creation counts when the response
        exposes them (Anthropic prompt caching), so the KV-cache hit
        rate can be tracked over time (paper 2504.15228 Table 1: SICA
        observed 31.9-40.9% cache hits across 15 iterations).
        """
        try:
            from neo.memory.metrics import record as metrics_record

            usage = getattr(response, "usage", None)
            if usage is None:
                return
            input_tokens = getattr(usage, "input_tokens", 0) or 0
            cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
            cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
            output_tokens = getattr(usage, "output_tokens", 0) or 0
            cacheable_input = input_tokens + cache_read + cache_creation
            cache_hit_rate = (
                cache_read / cacheable_input if cacheable_input > 0 else 0.0
            )
            metrics_record(
                "lm_call",
                provider="anthropic",
                model=self.model,
                status="success",
                input_tokens=input_tokens,
                cache_read_input_tokens=cache_read,
                cache_creation_input_tokens=cache_creation,
                output_tokens=output_tokens,
                cache_hit_rate=round(cache_hit_rate, 4),
            )
        except Exception:
            # Metrics are never load-bearing.
            pass

    def name(self) -> str:
        return f"anthropic/{self.model}"


# ============================================================================
# Google Adapter
# ============================================================================

class GoogleAdapter(LMAdapter):
    """Adapter for Google models (Gemini) using google-genai SDK."""

    def __init__(self, model: str = "gemini-2.0-flash", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key required")

        try:
            from google import genai
            # Create client with API key
            self.client = genai.Client(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "google-genai package required: pip install google-genai"
            )

    def generate(
        self,
        messages: list[dict[str, str]],
        stop: Optional[list[str]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: Optional[str] = None,  # not supported; accepted for ABC compat
    ) -> str:
        """Generate response using Google Generative AI SDK."""
        from google.genai import types

        # Convert messages to new SDK format
        # Messages use "user" or "model" roles, with content in "parts" array
        # Note: Google SDK requires "system" messages be mapped to "user" role
        # This is a known SDK limitation - system prompts are merged with user context
        formatted_messages = []
        for msg in messages:
            role = "user" if msg["role"] in ["user", "system"] else "model"
            formatted_messages.append({
                "role": role,
                "parts": [msg["content"]],
            })

        # Create generation config using types
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            stop_sequences=stop,
        )

        try:
            # Generate content using new SDK interface
            response = self.client.models.generate_content(
                model=self.model,
                contents=formatted_messages,
                config=config,
            )

            # Handle missing or None response text
            if not hasattr(response, 'text') or response.text is None:
                raise ValueError("API returned empty response")

            return response.text

        except Exception as e:
            error_msg = str(e).lower()

            # Handle common API errors with clear messages
            if "401" in error_msg or "403" in error_msg or "unauthorized" in error_msg:
                raise ValueError(f"Invalid API key: {e}")
            elif "429" in error_msg or "rate limit" in error_msg:
                raise ValueError(f"Rate limit exceeded: {e}")
            elif "404" in error_msg or "not found" in error_msg:
                raise ValueError(f"Invalid model '{self.model}': {e}")
            elif "network" in error_msg or "connection" in error_msg:
                raise ValueError(f"Network error: {e}")
            else:
                # Re-raise with original error for unexpected cases
                raise

    def name(self) -> str:
        return f"google/{self.model}"


# ============================================================================
# Local/OpenAI-Compatible Adapter
# ============================================================================

class LocalAdapter(LMAdapter):
    """Adapter for local models via OpenAI-compatible API (llama.cpp, vLLM, etc.)."""

    def __init__(
        self,
        model: str = "local-model",
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "not-needed",
    ):
        self.model = model
        self.base_url = base_url

        try:
            import openai
            self.client = openai.OpenAI(
                base_url=base_url,
                api_key=api_key,
            )
        except ImportError:
            raise ImportError("openai package required: pip install openai")

    def generate(
        self,
        messages: list[dict[str, str]],
        stop: Optional[list[str]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: Optional[str] = None,  # not supported; accepted for ABC compat
    ) -> str:
        """Generate response using local API."""
        # OpenAI-compatible endpoints front many providers (vLLM, llama.cpp,
        # xAI Grok, DeepSeek); resilient helper recovers if the model behind
        # the endpoint is a reasoning model that rejects `temperature` /
        # `max_tokens`.
        response = _chat_completion_resilient(self.client, {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop,
        }, provider="local")
        return response.choices[0].message.content

    def name(self) -> str:
        return f"local/{self.model}"


# ============================================================================
# Ollama Adapter
# ============================================================================

class OllamaAdapter(LMAdapter):
    """Adapter for Ollama-hosted models."""

    def __init__(
        self,
        model: str = "llama2",
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.base_url = base_url

        try:
            import requests
            self.requests = requests
        except ImportError:
            raise ImportError("requests package required: pip install requests")

    def generate(
        self,
        messages: list[dict[str, str]],
        stop: Optional[list[str]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: Optional[str] = None,  # not supported; accepted for ABC compat
    ) -> str:
        """Generate response using Ollama API."""
        # Convert messages to prompt
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"<|system|>\n{content}\n")
            elif role == "user":
                prompt_parts.append(f"<|user|>\n{content}\n")
            elif role == "assistant":
                prompt_parts.append(f"<|assistant|>\n{content}\n")

        prompt = "".join(prompt_parts) + "<|assistant|>\n"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if stop:
            payload["options"]["stop"] = stop

        response = self.requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
        )
        response.raise_for_status()
        return response.json()["response"]

    def name(self) -> str:
        return f"ollama/{self.model}"


# ============================================================================
# Azure OpenAI Adapter
# ============================================================================

class AzureOpenAIAdapter(LMAdapter):
    """Adapter for Azure OpenAI models."""

    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_version: str = "2024-02-15-preview",
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        self.endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.api_version = api_version

        if not self.api_key:
            raise ValueError("Azure OpenAI API key required")
        if not self.endpoint:
            raise ValueError("Azure OpenAI endpoint required")

        try:
            import openai
            self.client = openai.AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.endpoint,
                api_version=self.api_version,
            )
        except ImportError:
            raise ImportError("openai package required: pip install openai")

    def generate(
        self,
        messages: list[dict[str, str]],
        stop: Optional[list[str]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: Optional[str] = None,  # not supported; accepted for ABC compat
    ) -> str:
        """Generate response using Azure OpenAI API."""
        # Azure `model` is an arbitrary deployment name, so reasoning models
        # (gpt-5 / o-series deployments) can't be detected by string — they
        # reject `temperature` and require `max_completion_tokens` instead of
        # `max_tokens`. The resilient helper learns both from the 400.
        response = _chat_completion_resilient(self.client, {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop,
        }, provider="azure")
        return response.choices[0].message.content

    def name(self) -> str:
        return f"azure/{self.model}"


# ============================================================================
# Claude Code Adapter
# ============================================================================

class ClaudeCodeAdapter(LMAdapter):
    """
    Adapter that uses Claude Code CLI instead of direct Anthropic API.

    This allows Neo to leverage Claude Code Max/Pro subscriptions
    instead of incurring direct API billing costs.

    Requirements:
    - Claude Code CLI installed (`npm install -g @anthropic-ai/claude-code`)
    - User authenticated via `claude auth login`
    - ANTHROPIC_API_KEY should NOT be set (to force subscription auth)
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-5-20250929",
        cli_path: str = "claude",
        timeout: int = 600,
        api_key: Optional[str] = None,  # Accepted for compatibility, but not used
        base_url: Optional[str] = None,  # Accepted for compatibility, but not used
    ):
        self.model = model
        self.cli_path = cli_path
        self.timeout = timeout

        # Verify Claude Code CLI is available
        try:
            result = subprocess.run(
                [cli_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Claude Code CLI not found at: {cli_path}")
        except FileNotFoundError:
            raise RuntimeError(
                "Claude Code CLI not installed. "
                "Install with: npm install -g @anthropic-ai/claude-code"
            )

    def generate(
        self,
        messages: list[dict[str, str]],
        stop: Optional[list[str]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: Optional[str] = None,  # not supported; accepted for ABC compat
    ) -> str:
        """Generate response using Claude Code CLI."""

        # Extract system prompt from messages
        system_prompt = ""
        conversation_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                conversation_messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })

        # Write system prompt to temp file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.txt',
            delete=False
        ) as f:
            f.write(system_prompt)
            system_prompt_file = f.name

        try:
            # Build Claude Code CLI arguments
            args = [
                self.cli_path,
                "--system-prompt-file", system_prompt_file,
                "--output-format", "stream-json",
                "--model", self.model,
                "--max-turns", "1",
                "--verbose",
                "-p",
            ]

            # Prepare environment (remove API key to force subscription)
            env = {
                **os.environ,
                "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(max_tokens),
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            }
            env.pop("ANTHROPIC_API_KEY", None)

            # Run Claude Code CLI
            process = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
            )

            # Write messages to stdin
            process.stdin.write(json.dumps(conversation_messages))
            process.stdin.close()

            # Parse streaming JSON output
            response_text = ""
            for line in process.stdout:
                line = line.strip()
                if not line:
                    continue

                try:
                    chunk = json.loads(line)

                    if chunk.get("type") == "assistant":
                        message = chunk.get("message", {})
                        for content in message.get("content", []):
                            if content.get("type") == "text":
                                response_text += content.get("text", "")

                    if chunk.get("type") == "assistant":
                        message = chunk.get("message", {})
                        if message.get("stop_reason"):
                            content = message.get("content", [{}])[0]
                            if content.get("text", "").startswith("API Error"):
                                stderr = process.stderr.read()
                                raise RuntimeError(
                                    f"Claude Code error: {content['text']}\n{stderr}"
                                )

                except json.JSONDecodeError:
                    continue

            return_code = process.wait(timeout=self.timeout)

            if return_code != 0:
                stderr = process.stderr.read()
                raise RuntimeError(
                    f"Claude Code exited with code {return_code}: {stderr}"
                )

            return response_text.strip()

        finally:
            Path(system_prompt_file).unlink(missing_ok=True)

    def name(self) -> str:
        return f"claude-code/{self.model}"


# ============================================================================
# Adapter Factory
# ============================================================================

# ============================================================================
# CAR (Common Agent Runtime) Adapter — routes inference through car-server's
# unified inference layer (local Candle/MLX + remote OpenAI/Anthropic/Google
# behind one provider-agnostic protocol with Rust-enforced policies, adaptive
# routing, semantic conversation compaction, and deterministic replay).
# ============================================================================


#: Default for CAR's per-call FFI read timeout (``CAR_DAEMON_TIMEOUT``, seconds).
#: CAR ships a 30s default, but a quality remote serving neo's large multi-file
#: code context legitimately takes longer — measured ~140s for a 143 KB prompt
#: via parslee. At 30s those calls are cut off mid-inference and neo falls back
#: every time, so we raise the floor. Only applied when the operator hasn't set
#: ``CAR_DAEMON_TIMEOUT`` themselves. Kept strictly below the AutoAdapter
#: watchdog (``_CAR_CALL_TIMEOUT_S``) so CAR's own clean timeout fires first and
#: the watchdog only catches a true hang where even that doesn't return.
_CAR_DAEMON_TIMEOUT_DEFAULT_S = 180


def _apply_car_daemon_timeout_default() -> None:
    """Raise CAR's FFI read-timeout floor for neo's long-inference workload,
    unless the operator already set ``CAR_DAEMON_TIMEOUT``."""
    os.environ.setdefault("CAR_DAEMON_TIMEOUT", str(_CAR_DAEMON_TIMEOUT_DEFAULT_S))


def _model_pin_honored(requested: str, used: str) -> bool:
    """Whether CAR's ``model_used`` plausibly satisfies an explicit ``model`` pin.

    CAR expands a logical pin to a concrete deployment (a version/date suffix),
    so we accept when either name contains the other after stripping
    non-alphanumerics. A wholly unrelated ``used`` — e.g. a pin of
    ``nonexistent-model-id-xyz`` silently routed to the ``apple-foundation``
    fallback — means CAR ignored the pin; the caller should fail fast rather
    than run against the wrong model.

    Known limitation: a CAR-internal *alias* that resolves to an unrelated
    concrete name would false-positive here. neo pins concrete model names, and
    router mode (``model=None``) skips this check entirely, so that path isn't
    exercised in practice.
    """
    def _norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())

    r, u = _norm(requested), _norm(used)
    if not r or not u:
        return True  # nothing comparable — don't block
    return r in u or u in r


class CarAdapter(LMAdapter):
    """Route Neo's outbound inference through CAR's unified inference layer.

    With ``model=None`` (default), CAR's adaptive ``route_model`` picks local
    vs. remote per call using the supplied ``intent_hint``. Pin a backend by
    passing e.g. ``model='gpt-4.1-mini'`` or ``model='Qwen3-4B'``.

    Reuses the process-wide ``CarRuntime`` singleton in ``neo.car_inference``,
    so a single ``neo serve`` host and an outbound ``CarAdapter`` in the same
    process share state, policies, and the eventlog.

    CAR ``infer_tracked`` signature (validated against car-runtime 0.15.x)::

        infer_tracked(prompt, model=None, max_tokens=None, context=None,
                      tools_json=None, messages_json=None,
                      tool_choice=None, parallel_tool_calls=None,
                      intent_json=None) -> str

    Returns a JSON string with: ``text``, ``model_used``, ``usage``
    (``prompt_tokens`` / ``completion_tokens`` / ``total_tokens`` /
    ``context_window``), ``latency_ms``, ``time_to_first_token_ms``,
    ``trace_id``, ``tool_calls``. Note CAR does NOT accept ``temperature``
    via this API — the underlying backend's default is used.
    """

    #: Default IntentHint sent to CAR when the caller doesn't supply one.
    #: Neo's workload is overwhelmingly code reasoning (review, optimization,
    #: debugging, generation), so we tell the router (a) it's a code task and
    #: (b) ``prefer_quality`` — route to the most capable model, not the cheapest
    #: or fastest. This is what lets neo rely on CAR's router WITHOUT pinning a
    #: model version: the router picks the best available model (and, with CAR's
    #: auto-discovery, newly-released ones) under the quality workload. Without
    #: ``prefer_quality`` CAR's default profile is latency/cost-biased and
    #: routes neo to mini models — a measured quality regression. CAR task enum:
    #: ``chat | classify | reasoning | code``.
    DEFAULT_INTENT_HINT: dict = {"task": "code", "prefer_quality": True}

    def __init__(
        self,
        model: Optional[str] = None,
        intent_hint: Optional[dict] = None,
        runtime: Optional[object] = None,
    ):
        self.model = model
        # Caller-supplied intent wins; otherwise default to the coding workload.
        self.intent_hint = dict(intent_hint) if intent_hint else dict(self.DEFAULT_INTENT_HINT)
        # Give large-context inference room to finish before CAR's FFI read
        # timeout cuts it off (a quality remote can take ~140s on neo's context).
        _apply_car_daemon_timeout_default()
        if runtime is None:
            from neo.car_inference import get_runtime
            runtime = get_runtime()
        self._rt = runtime

    @staticmethod
    def _messages_to_prompt(messages) -> str:
        """Flatten a chat-style message list into a single prompt.

        Used as a fallback when we don't have a structured messages array;
        ``generate()`` prefers passing ``messages_json`` to CAR directly.
        """
        if isinstance(messages, str):
            return messages
        parts = []
        for m in messages:
            if isinstance(m, dict):
                role = m.get("role", "user")
                content = m.get("content", "")
                parts.append(f"{role}: {content}")
            else:
                parts.append(str(m))
        return "\n\n".join(parts)

    def generate(
        self,
        messages,
        stop: Optional[list] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        # ``temperature``, ``stop``, ``reasoning_effort`` are accepted for
        # LMAdapter ABC compatibility. CAR's infer_tracked does not expose
        # them as kwargs — backend defaults win. Future: translate
        # reasoning_effort∈{"high","xhigh"} to intent_hint["require"]=
        # ["reasoning"] so CAR's router escalates accordingly (see
        # github.com/Parslee-ai/car-releases/issues/52).
        kwargs: dict = {"max_tokens": int(max_tokens)}
        if self.model:
            kwargs["model"] = self.model
        if self.intent_hint is not None:
            # CAR expects intent_json as a JSON string, not a dict.
            kwargs["intent_json"] = json.dumps(self.intent_hint)

        # Prefer structured messages so CAR can apply its conversation
        # compaction and chat-template handling. Fall back to a flattened
        # prompt when the caller passed a bare string.
        from neo.memory.metrics import capture_lm_call_failure
        input_shape = (
            "string" if isinstance(messages, str)
            else "messages_list" if isinstance(messages, list)
            else type(messages).__name__
        )
        with capture_lm_call_failure(
            provider="car",
            model=self.model or "router",
            input_shape=input_shape,
            max_tokens=int(max_tokens),
            intent_task=(self.intent_hint or {}).get("task") if self.intent_hint else None,
        ):
            if isinstance(messages, str):
                result_raw = self._rt.infer_tracked(messages, **kwargs)
            elif isinstance(messages, list):
                kwargs["messages_json"] = json.dumps(messages)
                # Prompt arg is still required by infer_tracked; pass the first
                # user message (or an empty string) — messages_json takes priority.
                first_user = next(
                    (m.get("content", "") for m in messages if isinstance(m, dict) and m.get("role") == "user"),
                    "",
                )
                result_raw = self._rt.infer_tracked(first_user, **kwargs)
            else:
                result_raw = self._rt.infer_tracked(str(messages), **kwargs)

        try:
            result = json.loads(result_raw) if isinstance(result_raw, str) else result_raw
        except (ValueError, TypeError):
            # CAR violated its contract; surface the raw value as the text so
            # downstream parsers see something rather than crashing.
            return str(result_raw)

        try:
            from neo.memory.metrics import record as metrics_record
            usage = result.get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or 0)
            metrics_record(
                "lm_call",
                provider="car",
                model=result.get("model_used") or self.model or "router",
                status="success",
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                context_window=int(usage.get("context_window") or 0),
                latency_ms=int(result.get("latency_ms") or 0),
                time_to_first_token_ms=int(result.get("time_to_first_token_ms") or 0),
                trace_id=result.get("trace_id") or "",
            )
        except Exception:
            pass  # Metrics are never load-bearing.

        # Fail fast on a silently-substituted model. When a specific model is
        # pinned but CAR can't honor it (unknown/unavailable id), the router
        # falls back to a default instead of erroring — so a typo'd or retired
        # pin would run against the wrong model unnoticed. Router mode
        # (model=None) intentionally skips this.
        model_used = result.get("model_used") or ""
        if self.model and model_used and not _model_pin_honored(self.model, model_used):
            raise RuntimeError(
                f"CAR did not honor pinned model '{self.model}': it routed to "
                f"'{model_used}' instead. Unknown or unavailable model ids "
                f"silently fall back to a default — pin a model CAR knows, or "
                f"use router mode (model=None)."
            )

        return result.get("text", "")

    def name(self) -> str:
        return f"car/{self.model or 'router'}"


def create_adapter(
    provider: str,
    model: Optional[str] = None,
    **kwargs,
) -> LMAdapter:
    """
    Factory function to create appropriate adapter.

    Args:
        provider: One of "openai", "anthropic", "google", "azure", "local",
                  "ollama", "claude-code", "car"
        model: Model name (optional, uses provider default)
        **kwargs: Additional provider-specific arguments

    Returns:
        LMAdapter instance
    """
    provider = provider.lower()

    if provider == "openai":
        return OpenAIAdapter(model=model or "gpt-4", **kwargs)
    elif provider == "anthropic":
        return AnthropicAdapter(model=model or "claude-sonnet-4-5-20250929", **kwargs)
    elif provider == "google":
        return GoogleAdapter(model=model or "gemini-2.0-flash", **kwargs)
    elif provider == "azure":
        return AzureOpenAIAdapter(model=model or "gpt-4", **kwargs)
    elif provider == "local":
        return LocalAdapter(model=model or "local-model", **kwargs)
    elif provider == "ollama":
        return OllamaAdapter(model=model or "llama2", **kwargs)
    elif provider == "claude-code":
        return ClaudeCodeAdapter(model=model or "claude-sonnet-4-5-20250929", **kwargs)
    elif provider == "car":
        # CarAdapter ignores api_key/base_url — CAR's router owns provider
        # selection. Strip them so callers can pass NeoConfig values uniformly.
        kwargs.pop("api_key", None)
        kwargs.pop("base_url", None)
        return CarAdapter(model=model, **kwargs)
    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Supported: openai, anthropic, google, azure, local, ollama, claude-code, car"
        )


#: Seconds the breaker stays open after a CAR failure before half-opening to
#: re-probe CAR. Bounds retry-storms while letting a long-lived process recover
#: from a transient CAR outage instead of pinning to the fallback forever.
_CAR_RETRY_COOLDOWN_S = 300.0

#: Default per-call wall-clock deadline (seconds) for a single CAR inference
#: call in auto mode. ``car_runtime.infer_tracked`` is a blocking FFI call; if
#: the daemon is restarted mid-call (its PID churns on every CarHost relaunch)
#: the client's socket read can block with no deadline — observed once as a
#: 5-day hang of the whole neo process. Bounding the call turns that into a
#: clean failover to the static provider. Kept ABOVE the CAR FFI read timeout
#: (``_CAR_DAEMON_TIMEOUT_DEFAULT_S``) so CAR's own clean timeout normally fires
#: first (→ breaker → fallback) and this watchdog only catches a genuine hang
#: where even that never returns. Override with ``NEO_CAR_TIMEOUT_SECONDS``.
_CAR_CALL_TIMEOUT_S = 240.0


def _car_call_timeout() -> float:
    """Per-call CAR deadline, overridable via ``NEO_CAR_TIMEOUT_SECONDS``."""
    raw = os.environ.get("NEO_CAR_TIMEOUT_SECONDS")
    if raw:
        try:
            v = float(raw)
            if v > 0:
                return v
        except ValueError:
            pass
    return _CAR_CALL_TIMEOUT_S


class AutoAdapter(LMAdapter):
    """CAR-first adapter: route through CAR's dynamic router, fall back to a
    static adapter when a CAR call fails.

    Constructed only when CAR is usable at build time (see ``resolve_adapter``),
    so "CAR absent / daemon down at start" is handled there. This handles the
    other half — CAR failing mid-process (daemon dies, a call errors): a failure
    opens a circuit breaker for ``retry_cooldown`` seconds (no retry-storm),
    after which it half-opens and re-probes CAR (so a transient outage doesn't
    pin a long-lived observer/host to the fallback forever).

    The fallback is built **lazily** via ``fallback_factory`` — so when CAR is
    the only usable backend, a missing/invalid static-provider key never blocks
    CAR (it's only constructed if CAR actually fails).

    The breaker is intentionally lock-free: ``_disabled_until`` is a plain float
    written from ``generate``. Concurrent callers may both see CAR up, both
    fail, both push the cooldown — idempotent, no torn state — so car_host's
    shared per-project adapter is safe without a mutex.
    """

    def __init__(self, car: LMAdapter, fallback_factory,
                 retry_cooldown: float = _CAR_RETRY_COOLDOWN_S,
                 car_timeout: Optional[float] = None):
        self._car = car
        self._make_fallback = fallback_factory
        self._fallback: Optional[LMAdapter] = None
        self._retry_cooldown = retry_cooldown
        self._car_timeout = car_timeout if car_timeout is not None else _car_call_timeout()
        self._disabled_until = 0.0  # monotonic deadline; <= now => CAR eligible

    def _fallback_adapter(self) -> LMAdapter:
        if self._fallback is None:
            self._fallback = self._make_fallback()
        return self._fallback

    def _car_generate_bounded(self, messages, stop, max_tokens,
                              temperature, reasoning_effort) -> str:
        """Run the CAR call under a wall-clock deadline.

        ``infer_tracked`` is a blocking FFI call with no caller-visible
        timeout, so an error-only breaker can't catch a *hang* (a dead/
        restarted daemon leaving the socket read blocked forever). We run it
        in a daemon worker thread and abandon it on timeout, raising so the
        breaker opens and the next call routes to the fallback. The orphaned
        worker can't be force-killed, but it's a daemon (won't block process
        exit) and timeouts are rare + cooldown-bounded, so leaked threads
        can't accumulate unbounded.
        """
        box: dict = {}

        def _worker():
            try:
                box["value"] = self._car.generate(
                    messages, stop=stop, max_tokens=max_tokens,
                    temperature=temperature, reasoning_effort=reasoning_effort,
                )
            except BaseException as exc:  # noqa: BLE001 — relay to caller thread
                box["error"] = exc

        worker = threading.Thread(target=_worker, name="neo-car-call", daemon=True)
        worker.start()
        worker.join(self._car_timeout)
        if worker.is_alive():
            raise TimeoutError(
                f"CAR inference exceeded {self._car_timeout:.0f}s deadline"
            )
        if "error" in box:
            raise box["error"]
        return box.get("value", "")

    def generate(self, messages, stop=None, max_tokens: int = 4096,
                 temperature: float = 0.7, reasoning_effort: Optional[str] = None) -> str:
        if time.monotonic() >= self._disabled_until:
            try:
                return self._car_generate_bounded(
                    messages, stop, max_tokens, temperature, reasoning_effort,
                )
            except Exception as e:
                self._disabled_until = time.monotonic() + self._retry_cooldown
                logger.warning(
                    "CAR inference failed (%s); using the static fallback for ~%.0fs: %s",
                    type(e).__name__, self._retry_cooldown, e,
                )
        return self._fallback_adapter().generate(
            messages, stop=stop, max_tokens=max_tokens,
            temperature=temperature, reasoning_effort=reasoning_effort,
        )

    def name(self) -> str:
        return f"auto({self._car.name()} -> static)"


def _adapter_kwargs_for_config(config) -> dict:
    """Build provider-specific adapter kwargs from config."""
    provider = config.provider.lower()
    adapter_kwargs = {}
    if provider in ("openai", "anthropic", "google", "azure", "local", "claude-code"):
        adapter_kwargs["api_key"] = config.api_key
    if config.base_url:
        if provider in ("openai", "local", "ollama", "claude-code"):
            adapter_kwargs["base_url"] = config.base_url
        elif provider == "azure":
            adapter_kwargs["endpoint"] = config.base_url
    return adapter_kwargs


def resolve_adapter(config) -> LMAdapter:
    """Build the inference adapter for a NeoConfig, honoring ``inference_mode``.

    Default ("auto"): prefer CAR's dynamic router when car-runtime is importable
    AND the daemon is reachable, falling back to the configured static provider
    on absence or runtime failure. "static": always the configured provider.

    CAR is therefore optional — neo works without it — but used when present.
    The static adapter is built **lazily** (only when CAR is unavailable, or as
    the fallback after a CAR failure) so that a CAR-only install with no static
    provider key configured still works.
    """
    def build_static() -> LMAdapter:
        return create_adapter(
            config.provider,
            config.model,
            **_adapter_kwargs_for_config(config),
        )

    mode = str(getattr(config, "inference_mode", "auto") or "auto").strip().lower()
    if mode not in ("auto", "static"):
        logger.warning("unknown inference_mode %r; using 'auto'", mode)
        mode = "auto"

    if mode == "auto":
        # CAR-first: only build the CAR adapter when CAR is genuinely usable, so
        # we never pay an import/connect error just to fall back.
        try:
            from neo.a2ui import is_daemon_reachable
            from neo.car_inference import is_available as car_available
            if car_available() and is_daemon_reachable():
                car = create_adapter("car", model=None)  # model=None -> CAR routes dynamically
                return AutoAdapter(car, build_static)
        except Exception as e:
            logger.debug("CAR-first unavailable, using static provider: %s", e)
    return build_static()
