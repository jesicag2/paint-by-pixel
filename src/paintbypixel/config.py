"""Config loading and prompt building."""
from __future__ import annotations

import copy
import csv
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Load the default config, then an optional config file, then overrides."""
    with open(DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    if path and Path(path).resolve() != DEFAULT_CONFIG:
        with open(path) as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})
    return _deep_merge(cfg, overrides or {})


def build_prompt(cfg: dict, subject: str) -> str:
    """Wrap a plain subject ("strawberry shortcake slice") in the shared template."""
    return cfg["generation"]["prompt_template"].format(prompt=subject.strip())


def resolve(path: str | Path) -> Path:
    """Resolve a config path relative to the repo root."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def read_prompts(cfg: dict) -> list[dict]:
    """Read the held-out test prompts: rows with id, category, subject."""
    with open(resolve(cfg["eval"]["prompts_file"]), newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["id"] = int(r["id"])
    return rows
