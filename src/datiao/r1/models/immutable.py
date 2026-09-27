"""Helpers for recursively immutable JSON payloads."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any


class FrozenDict(dict[str, Any]):
    """A JSON-serializable dict that rejects normal mutation operations."""

    def __init__(self, values: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        dict.__init__(self, values or {}, **kwargs)

    def _immutable(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("source_payload is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable


def freeze_json(value: Any) -> Any:
    """Recursively convert a JSON-compatible value to immutable containers."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("source_payload cannot contain non-finite numbers")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("source_payload object keys must be strings")
        return FrozenDict({key: freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item) for item in value)
    raise TypeError(f"unsupported source_payload value: {type(value).__name__}")


def thaw_json(value: Any) -> Any:
    """Return regular JSON containers for canonical hashing or transport."""

    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


def stable_json_hash(value: Any) -> str:
    canonical = json.dumps(
        thaw_json(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
