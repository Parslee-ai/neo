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
    model: Optional[str] = "gpt-5-codex"
    api_key: Optional[str] = None
    base_url: Optional[str] = None  # For local/ollama

    # Generation settings
    default_temperature: float = 0.7
    default_max_tokens: int = 4096

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

        return cls(**filtered_data)

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
        }

        with open(path, "w") as f:
            json.dump(exposed_fields, f, indent=2)