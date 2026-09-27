"""R1 output policy and forbidden inference fields."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

OUT_OF_SCOPE_FIELDS = frozenset(
    {"emotion", "attention_score", "careless", "wrong_reason", "knowledge_difficulty"}
)


def reject_out_of_scope_fields(value: Any, path: str = "metadata") -> Any:
    """Reject forbidden inference keys anywhere in event metadata."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in OUT_OF_SCOPE_FIELDS:
                raise ValueError(f"out-of-scope field: {path}.{key}")
            reject_out_of_scope_fields(item, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            reject_out_of_scope_fields(item, f"{path}[{index}]")
    return value
