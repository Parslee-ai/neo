"""
Helper to load environment variables from .env file.

Usage:
    from load_env import load_env
    load_env()
"""

import os
from pathlib import Path


def load_env(env_file: str = ".env"):
    """
    Load environment variables from .env file.

    Args:
        env_file: Path to .env file (default: .env in current directory)
    """
    env_path = Path(env_file)

    if not env_path.exists():
        # Try relative to script location
        script_dir = Path(__file__).parent
        env_path = script_dir / env_file

    if not env_path.exists():
        return  # No .env file, skip silently

    with open(env_path) as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            # Parse KEY=VALUE
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()

                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]

                # Set environment variable (don't override existing)
                if key and not os.environ.get(key):
                    os.environ[key] = value


# Auto-load when imported
if __name__ != "__main__":
    load_env()