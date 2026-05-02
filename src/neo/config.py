"""
Configuration management for Neo.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class NeoConfig:
    """Neo configuration."""

    # LM Provider settings
    provider: str = "openai"  # openai, anthropic, google, azure, local, ollama
    model: Optional[str] = "gpt-5.5"
    api_key: Optional[str] = None
    base_url: Optional[str] = None  # For local/ollama

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

        with open(path) as f:
            data = json.load(f)

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

        # API keys
        config.api_key = (
            os.environ.get("NEO_API_KEY") or
            os.environ.get("OPENAI_API_KEY") or
            os.environ.get("ANTHROPIC_API_KEY") or
            os.environ.get("GOOGLE_API_KEY")
        )

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

        # Override with environment variables
        env_config = cls.from_env()
        for key, value in env_config.__dict__.items():
            if value is not None and value != getattr(cls(), key):
                setattr(config, key, value)

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
            'api_key': self.api_key,
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

        with open(path, "w") as f:
            json.dump(exposed_fields, f, indent=2)