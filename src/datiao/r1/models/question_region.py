"""Question region model and geometry validation."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Real
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class QuestionRegion(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    region_id: str = Field(min_length=1)
    page_id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    geometry_type: Literal["rectangle", "polygon"]
    coordinates: tuple[float, ...]
    priority: int = 0

    @field_validator("coordinates", mode="before")
    @classmethod
    def normalize_coordinates(cls, value: object) -> object:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            return value
        if any(isinstance(item, bool) or not isinstance(item, Real) for item in value):
            raise ValueError("coordinates must contain finite numbers")
        return tuple(float(item) for item in value)

    @model_validator(mode="after")
    def validate_geometry(self) -> "QuestionRegion":
        if self.geometry_type == "rectangle":
            if len(self.coordinates) != 4:
                raise ValueError("rectangle coordinates must be (x, y, width, height)")
            if self.coordinates[2] < 0 or self.coordinates[3] < 0:
                raise ValueError("rectangle width and height must be non-negative")
        elif len(self.coordinates) < 6 or len(self.coordinates) % 2:
            raise ValueError("polygon coordinates must contain at least three points")
        return self

    def contains(self, x: float, y: float) -> bool:
        """Return whether a point lies inside this region, including its boundary."""

        if self.geometry_type == "rectangle":
            left, top, width, height = self.coordinates
            return left <= x <= left + width and top <= y <= top + height

        vertices = tuple(
            zip(self.coordinates[::2], self.coordinates[1::2], strict=True)
        )
        inside = False
        for index, (x1, y1) in enumerate(vertices):
            x2, y2 = vertices[(index + 1) % len(vertices)]
            if _point_on_segment(x, y, x1, y1, x2, y2):
                return True
            if (y1 > y) != (y2 > y):
                crossing_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
                if x < crossing_x:
                    inside = not inside
        return inside


def _point_on_segment(
    x: float,
    y: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> bool:
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if abs(cross) > 1e-9:
        return False
    return min(x1, x2) - 1e-9 <= x <= max(x1, x2) + 1e-9 and min(
        y1, y2
    ) - 1e-9 <= y <= max(y1, y2) + 1e-9
