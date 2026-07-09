"""Pytest configuration. Load .env.local for tests."""
import os
from pathlib import Path

# Load .env.local if it exists
env_local = Path(__file__).parent.parent / ".env.local"
if env_local.exists():
    with open(env_local) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key] = val
