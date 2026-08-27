"""
Configuration management for Neo.
"""

import json
import logging
import os
import platform
import shutil
import subprocess
import tempfile
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Optional


KEYCHAIN_SERVICE_PREFIX = "neo-reasoner"
logger = logging.getLogger(__name__)

#: Fields the generic save path must never write. `save` handles `api_key`
#: explicitly, gated on NEO_ALLOW_PLAINTEXT_API_KEY. Anything added here needs
#: the same treatment or it will simply stop being persisted.
_SECRET_FIELDS = frozenset({"api_key"})


#: Provider -> the environment variable that carries its key. Single-sourced:
#: this was duplicated at both resolution sites, and a third copy was about to
#: be added for the CAR fallback below.
PROVIDER_ENV_VAR = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "azure": "AZURE_OPENAI_API_KEY",
}

#: What CAR calls the same key, where it differs. CAR names the Gemini key
#: `GEMINI_API_KEY`; neo calls the provider `google`. Anything absent here uses
#: `PROVIDER_ENV_VAR`, and both names are tried, so a key stored under either
#: is found.
_CAR_KEY_ALIASES = {"google": "GEMINI_API_KEY"}

_CAR_SECRET_SERVICE = "car"
_CAR_LOOKUP_TIMEOUT_SECONDS = 5.0


def _keychain_service(provider: str) -> str:
    return f"{KEYCHAIN_SERVICE_PREFIX}:{provider}:api_key"


def _car_key_names(provider: str) -> list:
    """Names CAR might hold this provider's key under, in preference order."""
    names = []
    for name in (_CAR_KEY_ALIASES.get(provider.lower()),
                 PROVIDER_ENV_VAR.get(provider.lower())):
        if name and name not in names:
            names.append(name)
    return names


def load_api_key_from_car(provider: str) -> Optional[str]:
    """Read this provider's key from CAR's secret store, if CAR is installed.

    Neo already treats CAR as a first-class backend — `car-runtime` is an
    extra, `CarAdapter` is a provider, `neo serve` hosts on car-server, and the
    observer runs under CAR's supervisor. On a machine where CAR holds the
    credential, neo refusing to look was an oversight rather than a boundary:
    the two stores use disjoint names for the same key in the same keychain
    (`car`/`OPENAI_API_KEY` against `neo-reasoner:openai:api_key`), so both
    tools truthfully reported "no key" while the key sat between them.

    Delegates to the `car` CLI rather than reading the keychain directly. That
    keeps CAR's naming in CAR, and it is the only portable route: neo's own
    `keychain_available()` is `platform.system() == "Darwin"`, while CAR's store
    covers Keychain, Credential Manager and Secret Service. So this also gives
    neo credential storage on Windows and Linux, which it has none of today.

    Last in the chain by design — it forks a subprocess, so it must never run
    when an env var or neo's own keychain already answered. Quiet on every
    failure: no CAR, no entry, a timeout, a broken install. Opt out with
    `NEO_CAR_SECRETS=0`.
    """
    if not provider or os.environ.get("NEO_CAR_SECRETS") == "0":
        return None
    car = shutil.which("car")
    if not car:
        return None
    for name in _car_key_names(provider):
        try:
            result = subprocess.run(
                [car, "secrets", "get", "--service", _CAR_SECRET_SERVICE, name],
                check=False,
                capture_output=True,
                text=True,
                timeout=_CAR_LOOKUP_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode == 0:
            # `secrets get` prints the value with a trailing newline; the stored
            # value itself must not be assumed to carry one either way.
            key = result.stdout.strip()
            if key:
                return key
    return None


def keychain_available() -> bool:
    """Return True when the platform has the macOS security CLI."""
    return platform.system() == "Darwin"


# `security` prompts for keychain unlock when the keychain is locked, and an
# unattended run has nobody to answer it. Unbounded, that is an indefinite hang
# during config load — before neo does any work at all.
KEYCHAIN_TIMEOUT_SECONDS = 10


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
            timeout=KEYCHAIN_TIMEOUT_SECONDS,
        )
    except OSError:
        return None
    except subprocess.TimeoutExpired:
        logger.warning(
            "Keychain lookup timed out after %ss - the keychain may be locked; "
            "no API key was read", KEYCHAIN_TIMEOUT_SECONDS,
        )
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

    # Bounded, and the TimeoutExpired is deliberately NOT caught: this
    # function's contract is "the key is stored or you hear about it". A
    # timeout means we do not know whether it was written, and swallowing that
    # would report a durable secret that may not exist.
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
        timeout=KEYCHAIN_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise RuntimeError(f"Failed to store API key in Keychain: {detail}")


@dataclass
class NeoConfig:
    """Neo configuration."""

    # LM Provider settings
    provider: str = "openai"  # openai, anthropic, google, azure, local, ollama
    model: Optional[str] = "gpt-5.6"
    api_key: Optional[str] = None
    base_url: Optional[str] = None  # For local/ollama

    # Inference routing.
    #   "static" — always use the configured `provider` (never CAR). (default)
    #   "auto"   — prefer CAR's dynamic router when car-runtime is importable AND
    #              the daemon is reachable; fall back to the static provider above
    #              on absence or runtime failure. CAR is optional but used when
    #              present.
    # Default is "static" (gpt-5.6) until a CAR release verifies the router's
    # quality behavior — CAR's released router cost-biases to mini models, a
    # measured regression. Flip to "auto" once a verified CAR build is deployed.
    inference_mode: str = "static"

    # Reasoning tier: "auto" gates multi-agent deliberation on novelty + CAR +
    # at least one capable model (docs/solutions/tiered-reasoning-multi-agent.md;
    # a controlled A/B/A found the panel's win is the orchestration structure,
    # which holds same-model, so no diverse pool is required); "fast" forces the
    # single-call path; "deep" forces deliberation (degrades to a high-effort
    # single pass when CAR isn't available).
    reasoning_mode: str = "auto"  # "auto" | "fast" | "deep"

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
        provider_key = PROVIDER_ENV_VAR.get(config.provider.lower())
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

        provider_key_env = PROVIDER_ENV_VAR.get(config.provider.lower())
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
        if not config.api_key:
            # Last: CAR's store. Forks a subprocess, so it runs only once
            # everything cheaper has missed.
            config.api_key = load_api_key_from_car(config.provider)

        return config

    def _changed_from_defaults(self) -> dict:
        """Fields whose value differs from the dataclass default.

        Derived from the dataclass rather than a hand-maintained allow-list.
        The old list-based approach silently erased any field it omitted,
        because ``from_file`` accepts *every* field the class defines: a value
        could be hand-edited into config.json, read back correctly, then
        dropped by the next unrelated ``--config set``. That happened to
        ``inference_mode`` and ``reasoning_mode``, and would have happened to
        the six other fields nothing writes yet.

        Fields still at their default are omitted rather than written out.
        Persisting defaults would freeze today's values into every user's file,
        so a future change to a default would never reach anyone who had ever
        run ``--config set`` — the exact failure the ``auto_install_updates``
        migration in ``from_file`` exists to undo.

        ``api_key`` is excluded here and handled explicitly by ``save``: it is
        the one secret, and ``load()`` populates it from the environment or
        Keychain, so a generic diff-from-default pass would spill it to disk.
        """
        changed = {}
        for f in fields(self):
            if f.name in _SECRET_FIELDS:
                continue
            if f.default is not MISSING:
                default = f.default
            elif f.default_factory is not MISSING:
                default = f.default_factory()
            else:
                changed[f.name] = getattr(self, f.name)
                continue
            value = getattr(self, f.name)
            if value != default:
                changed[f.name] = value
        return changed

    def save(self, config_path: Optional[str] = None):
        """Save configuration to file (user-set fields only)."""
        if not config_path:
            config_dir = Path.home() / ".neo"
            config_dir.mkdir(exist_ok=True)
            config_path = str(config_dir / "config.json")

        path = Path(config_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)

        exposed_fields = self._changed_from_defaults()

        # The secret. Written unconditionally — as an explicit null when
        # plaintext storage is not enabled — so that a stale plaintext key
        # left in the file is actively cleared rather than merely omitted.
        exposed_fields['api_key'] = (
            self.api_key if os.environ.get("NEO_ALLOW_PLAINTEXT_API_KEY") else None
        )

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
