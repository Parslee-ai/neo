"""
Configuration management for Neo.
"""

import json
import logging
import os
import platform
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


KEYCHAIN_SERVICE_PREFIX = "neo-reasoner"
logger = logging.getLogger(__name__)


def _keychain_service(provider: str) -> str:
    return f"{KEYCHAIN_SERVICE_PREFIX}:{provider}:api_key"


def keychain_available() -> bool:
    """Return True when the platform has the macOS security CLI."""
    return platform.system() == "Darwin"


def load_api_key_from_keychain(provider: str) -> Optional[str]:
    """Load a provider API key from macOS Keychain, if available."""
    if not provider or not keychain_available():
        return None

    try:
        result = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-s",
                _keychain_service(provider),
                "-a",
                provider,
                "-w",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None

    if result.returncode != 0:
        return None
    key = result.stdout.strip()
    return key or None


def store_api_key_in_keychain(provider: str, api_key: str) -> None:
    """Persist a provider API key in macOS Keychain."""
    if not provider:
        raise ValueError("Provider is required to store an API key")
    if not api_key:
        raise ValueError("API key is required")
    if not keychain_available():
        raise RuntimeError("Durable secret storage is only implemented for macOS Keychain")

    result = subprocess.run(
        [
            "security",
            "add-generic-password",
            "-U",
            "-s",
            _keychain_service(provider),
            "-a",
            provider,
            "-w",
            api_key,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise RuntimeError(f"Failed to store API key in Keychain: {detail}")


@dataclass
class NeoConfig:
    """Neo configuration."""

    # LM Provider settings
    provider: str = "openai"  # openai, anthropic, google, azure, local, ollama
    model: Optional[str] = "gpt-5.5"
    api_key: Optional[str] = None
    base_url: Optional[str] = None  # For local/ollama

    # Inference routing.
    #   "static" — always use the configured `provider` (never CAR). (default)
    #   "auto"   — prefer CAR's dynamic router when car-runtime is importable AND
    #              the daemon is reachable; fall back to the static provider above
    #              on absence or runtime failure. CAR is optional but used when
    #              present.
    # Default is "static" (gpt-5.5) until a CAR release verifies the router's
    # quality behavior — CAR's released router cost-biases to mini models, a
    # measured regression. Flip to "auto" once a verified CAR build is deployed.
    inference_mode: str = "static"

    # Generation settings
    default_temperature: float = 0.7
    default_max_tokens: int = 4096

    # Reasoning effort (OpenAI gpt-5* only). Acts as an upper bound on the
    # memory-driven effort selected per-query. None = no cap.
    # Valid: "none", "low", "medium", "high", "xhigh".
    reasoning_effort_cap: Optional[str] = None

    # Safety settings
    safe_read_patterns: list[str] = field(default_factory=lambda: [
        "*.py", "*.js", "*.ts", "*.go", "*.rs", "*.java", "*.cpp", "*.c", "*.h",
        "*.md", "*.txt", "*.json", "*.yaml", "*.yml", "*.toml",
    ])
    forbidden_paths: list[str] = field(default_factory=lambda: [
        ".env", "*.key", "*.pem", "*.secret", "*credentials*",
    ])

    # Exemplar storage
    exemplar_dir: Optional[str] = None

    # Static analysis tools
    enable_ruff: bool = True
    enable_pyright: bool = True
    enable_mypy: bool = False
    enable_eslint: bool = True

    # Auto-update settings
    auto_install_updates: bool = True  # Automatically install updates in background

    # Memory backend settings
    memory_backend: str = "fact_store"  # "fact_store" (new) or "legacy" (PersistentReasoningMemory)
    constraint_auto_scan: bool = True  # Auto-scan CLAUDE.md etc. for constraints

    # Logging settings
    log_level: str = "WARNING"  # DEBUG, INFO, WARNING, ERROR

    def __post_init__(self) -> None:
        # Validate reasoning_effort_cap up-front so a typo fails at config
        # load rather than burning an API round-trip with `unsupported_value`.
        from neo.reasoning_effort import validate_effort
        self.reasoning_effort_cap = validate_effort(self.reasoning_effort_cap)

    @classmethod
    def from_file(cls, config_path: str) -> "NeoConfig":
        """Load configuration from JSON file."""
        path = Path(config_path).expanduser()
        if not path.exists():
            return cls()

        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"Failed to load config from {path}: {exc}; using defaults")
            return cls()

        # Filter out fields that no longer exist (backward compatibility)
        import inspect
        valid_fields = set(inspect.signature(cls).parameters.keys())
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        config = cls(**filtered_data)

        # One-time migration: auto_install_updates default changed to True
        # in 0.13.1. Old configs saved False as the default. Flip it unless
        # the user explicitly opted out (marked by _auto_update_explicit).
        if (data.get("auto_install_updates") is False
                and "_auto_update_explicit" not in data):
            config.auto_install_updates = True

        return config

    @classmethod
    def from_env(cls) -> "NeoConfig":
        """Load configuration from environment variables."""
        config = cls()

        # Provider settings
        if provider := os.environ.get("NEO_PROVIDER"):
            config.provider = provider
        if model := os.environ.get("NEO_MODEL"):
            config.model = model
        if base_url := os.environ.get("NEO_BASE_URL"):
            config.base_url = base_url
        if mode := os.environ.get("NEO_INFERENCE_MODE"):
            config.inference_mode = mode

        # API keys. NEO_API_KEY is the explicit generic override; otherwise
        # choose the provider-specific key for the selected provider only.
        provider_key_env = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "azure": "AZURE_OPENAI_API_KEY",
        }
        provider_key = provider_key_env.get(config.provider.lower())
        config.api_key = os.environ.get("NEO_API_KEY")
        if config.api_key is None and provider_key:
            config.api_key = os.environ.get(provider_key)

        # Generation settings
        if temp := os.environ.get("NEO_TEMPERATURE"):
            config.default_temperature = float(temp)
        if max_tok := os.environ.get("NEO_MAX_TOKENS"):
            config.default_max_tokens = int(max_tok)

        # Exemplar storage
        if exemplar_dir := os.environ.get("NEO_EXEMPLAR_DIR"):
            config.exemplar_dir = exemplar_dir

        # Auto-update settings
        if auto_install := os.environ.get("NEO_AUTO_INSTALL_UPDATES"):
            config.auto_install_updates = auto_install.lower() in ("1", "true", "yes")

        # Logging settings
        if log_level := os.environ.get("NEO_LOG_LEVEL"):
            config.log_level = log_level.upper()

        # Reasoning effort cap (validated by __post_init__ via assignment? no —
        # __post_init__ ran on construction. Re-validate explicitly.)
        if effort := os.environ.get("NEO_REASONING_EFFORT"):
            from neo.reasoning_effort import validate_effort
            config.reasoning_effort_cap = validate_effort(effort)

        return config

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "NeoConfig":
        """
        Load configuration with priority:
        1. Explicit config file (if provided)
        2. ~/.neo/config.json
        3. Environment variables
        4. Defaults
        """
        if config_path:
            return cls.from_file(config_path)

        # Try default config location
        default_path = Path.home() / ".neo" / "config.json"
        if default_path.exists():
            config = cls.from_file(str(default_path))
        else:
            config = cls()
        original_provider = config.provider

        # Override with environment variables. Check env var presence rather
        # than comparing against class defaults: users must be able to
        # explicitly reset a saved config back to a default value, e.g.
        # NEO_PROVIDER=openai over a saved provider=anthropic.
        env_config = cls.from_env()
        env_overrides = {
            "NEO_PROVIDER": "provider",
            "NEO_MODEL": "model",
            "NEO_BASE_URL": "base_url",
            "NEO_INFERENCE_MODE": "inference_mode",
            "NEO_TEMPERATURE": "default_temperature",
            "NEO_MAX_TOKENS": "default_max_tokens",
            "NEO_EXEMPLAR_DIR": "exemplar_dir",
            "NEO_AUTO_INSTALL_UPDATES": "auto_install_updates",
            "NEO_LOG_LEVEL": "log_level",
            "NEO_REASONING_EFFORT": "reasoning_effort_cap",
        }
        for env_name, field_name in env_overrides.items():
            if env_name in os.environ:
                setattr(config, field_name, getattr(env_config, field_name))

        provider_key_env = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "azure": "AZURE_OPENAI_API_KEY",
        }.get(config.provider.lower())
        if "NEO_API_KEY" in os.environ or (
            provider_key_env is not None and provider_key_env in os.environ
        ):
            config.api_key = env_config.api_key
        elif config.provider != original_provider:
            # A provider override must not reuse a plaintext key saved for the
            # previous provider. Fall through to provider-specific Keychain.
            config.api_key = None

        if not config.api_key:
            config.api_key = load_api_key_from_keychain(config.provider)

        return config

    def save(self, config_path: Optional[str] = None):
        """Save configuration to file (only exposed fields)."""
        if not config_path:
            config_dir = Path.home() / ".neo"
            config_dir.mkdir(exist_ok=True)
            config_path = str(config_dir / "config.json")

        path = Path(config_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)

        # Only save exposed fields (not internal settings)
        exposed_fields = {
            'provider': self.provider,
            'model': self.model,
            'api_key': self.api_key if os.environ.get("NEO_ALLOW_PLAINTEXT_API_KEY") else None,
            'base_url': self.base_url,
            'auto_install_updates': self.auto_install_updates,
            'memory_backend': self.memory_backend,
            'constraint_auto_scan': self.constraint_auto_scan,
            'log_level': self.log_level,
            'reasoning_effort_cap': self.reasoning_effort_cap,
        }

        # Mark explicit opt-out so migration doesn't override it
        if self.auto_install_updates is False:
            exposed_fields['_auto_update_explicit'] = True

        fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(exposed_fields, f, indent=2)
            os.replace(tmp_name, path)
        except BaseException:
            os.unlink(tmp_name)
            raise
