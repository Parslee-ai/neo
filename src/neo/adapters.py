"""
LM Adapter implementations for OpenAI, Anthropic, and local models.
"""

import os
from typing import Optional

# Load environment variables from .env file
try:
    from neo.load_env import load_env
    load_env()
except ImportError:
    pass

from neo.cli import LMAdapter


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
    ) -> str:
        """Generate response using OpenAI API."""
        # gpt-5-codex uses /v1/responses endpoint
        if "codex" in self.model.lower() or "gpt-5" in self.model.lower():
            import httpx
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            base_url = self.base_url or "https://api.openai.com"
            url = f"{base_url}/v1/responses"

            payload = {
                "model": self.model,
                "input": messages,
            }

            response = httpx.post(url, headers=headers, json=payload, timeout=600.0)  # 10 minutes for complex queries
            if response.status_code != 200:
                raise ValueError(f"API error {response.status_code}: {response.text}")
            data = response.json()

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
            # Standard chat completions for other models
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
            )
            return response.choices[0].message.content

    def name(self) -> str:
        return f"openai/{self.model}"


# ============================================================================
# Anthropic Adapter
# ============================================================================

class AnthropicAdapter(LMAdapter):
    """Adapter for Anthropic models (Claude)."""

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
            "temperature": temperature,
        }

        if system_message:
            kwargs["system"] = system_message

        if stop:
            kwargs["stop_sequences"] = stop

        response = self.client.messages.create(**kwargs)
        return response.content[0].text

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
    ) -> str:
        """Generate response using local API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )
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
    ) -> str:
        """Generate response using Azure OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )
        return response.choices[0].message.content

    def name(self) -> str:
        return f"azure/{self.model}"


# ============================================================================
# Adapter Factory
# ============================================================================

def create_adapter(
    provider: str,
    model: Optional[str] = None,
    **kwargs,
) -> LMAdapter:
    """
    Factory function to create appropriate adapter.

    Args:
        provider: One of "openai", "anthropic", "google", "azure", "local", "ollama"
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
    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Supported: openai, anthropic, google, azure, local, ollama"
        )