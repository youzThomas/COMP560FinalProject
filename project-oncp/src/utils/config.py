"""Lightweight config loader with attribute-style access."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


class Config(dict):
    """Dict that also supports attribute access and recursive wrapping."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        if isinstance(value, Mapping) and not isinstance(value, Config):
            value = Config(value)
            self[key] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _wrap(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return Config({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def load_config(path: str | Path) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _wrap(raw or {})
