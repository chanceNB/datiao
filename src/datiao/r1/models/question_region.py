"""Question region V1 polygon contract and legacy geometry compatibility."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Real
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class QuestionRegion(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    schema_version: str = Field(default="1.0.0", min_length=1)
    region_id: str = Field(min_length=1)
    page_id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    region_type: Literal["rectangle", "polygon"]
    polygon_norm: tuple[tuple[float, float], ...]
    priority: int = 0
    coordinate_space: Literal["norm", "legacy"] = "norm"

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_shape(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        values = dict(data)
        if "geometry_type" in values:
            geometry_type = values.pop("geometry_type")
            coordinates = values.pop("coordinates", ())
            if not isinstance(coordinates, Sequence) or isinstance(coordinates, (str, bytes, bytearray)):
                return values
            numbers = tuple(float(item) for item in coordinates)
            if geometry_type == "rectangle" and len(numbers) != 4:
                raise ValueError("rectangle coordinates must be (x, y, width, height)")
            if geometry_type == "rectangle" and len(numbers) == 4:
                x, y, width, height = numbers
                polygon = ((x, y), (x + width, y), (x + width, y + height), (x, y + height))
            else:
                polygon = tuple(zip(numbers[::2], numbers[1::2], strict=True))
            values["region_type"] = geometry_type
            values["polygon_norm"] = polygon
            values["coordinate_space"] = "legacy"
        if isinstance(values.get("polygon_norm"), list):
            values["polygon_norm"] = tuple(tuple(point) for point in values["polygon_norm"])
        return values

    @model_validator(mode="after")
    def validate_geometry(self) -> "QuestionRegion":
        if len(self.polygon_norm) < 3:
            raise ValueError("polygon_norm must contain at least three points")
        if self.region_type == "rectangle" and len(self.polygon_norm) != 4:
            raise ValueError("rectangle polygon_norm must contain four points")
        for point in self.polygon_norm:
            if len(point) != 2 or any(isinstance(value, bool) or not isinstance(value, Real) for value in point):
                raise ValueError("polygon_norm points must contain finite numbers")
            if self.coordinate_space == "norm" and any(value < 0 or value > 1 for value in point):
                raise ValueError("polygon_norm coordinates must be between 0 and 1")
        return self

    @property
    def geometry_type(self) -> str:
        return self.region_type

    @property
    def coordinates(self) -> tuple[float, ...]:
        return tuple(value for point in self.polygon_norm for value in point)

    def contains(self, x: float, y: float) -> bool:
        vertices = self.polygon_norm
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


def _point_on_segment(x: float, y: float, x1: float, y1: float, x2: float, y2: float) -> bool:
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if abs(cross) > 1e-9:
        return False
    return min(x1, x2) - 1e-9 <= x <= max(x1, x2) + 1e-9 and min(y1, y2) - 1e-9 <= y <= max(y1, y2) + 1e-9
