"""Load client configuration from config.yaml and the .env file."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv


# ── Defaults (used when a key is missing from config.yaml) ────────────────────

_DEFAULTS: dict[str, Any] = {
    "server": {"url": "http://localhost:8000", "timeout": 120},
    "model": {"enable_thinking": True},
    "generation": {
        "max_tokens": 512,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "repeat_penalty": 1.0,
    },
    "conversation": {"max_history_messages": 10},
}


@dataclass
class GenerationDefaults:
    """Default generation/sampling parameters."""

    max_tokens: int = 512
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    repeat_penalty: float = 1.0


@dataclass
class ClientConfig:
    """Fully-resolved client configuration."""

    server_url: str = "http://localhost:8000"
    refresh_token: str = "teddy#1"
    timeout: int = 120
    enable_thinking: bool = True
    generation: GenerationDefaults = field(default_factory=GenerationDefaults)
    max_history_messages: int = 10


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (non-destructive)."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _find_project_root() -> Path:
    """Walk upward from this file to find the directory containing config.yaml."""
    current = Path(__file__).resolve().parent
    for ancestor in [current, *current.parents]:
        if (ancestor / "config.yaml").exists():
            return ancestor
    return current.parent


def load_config(config_path: Optional[str | Path] = None) -> ClientConfig:
    """Build a ``ClientConfig`` from *config.yaml* and environment variables.

    Parameters
    ----------
    config_path:
        Explicit path to a YAML config file.  When *None* the loader
        searches upward from this source file for ``config.yaml``.
    """
    root = _find_project_root()

    # Load .env (if present) so INFERENCE_REFRESH_TOKEN is available via os.environ
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    # Read YAML
    if config_path is None:
        config_path = root / "config.yaml"
    else:
        config_path = Path(config_path)

    raw: dict[str, Any] = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

    cfg = _deep_merge(_DEFAULTS, raw)

    # Build generation defaults
    gen_section = cfg.get("generation", {})
    gen = GenerationDefaults(
        max_tokens=int(gen_section.get("max_tokens", 512)),
        temperature=float(gen_section.get("temperature", 0.6)),
        top_p=float(gen_section.get("top_p", 0.95)),
        top_k=int(gen_section.get("top_k", 20)),
        min_p=float(gen_section.get("min_p", 0.0)),
        repeat_penalty=float(gen_section.get("repeat_penalty", 1.0)),
    )

    refresh_token = os.environ.get("INFERENCE_REFRESH_TOKEN", "")

    return ClientConfig(
        server_url=cfg["server"]["url"].rstrip("/"),
        refresh_token=refresh_token,
        timeout=int(cfg["server"]["timeout"]),
        enable_thinking=bool(cfg["model"]["enable_thinking"]),
        generation=gen,
        max_history_messages=int(cfg["conversation"]["max_history_messages"]),
    )
