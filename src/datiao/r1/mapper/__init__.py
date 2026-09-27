"""Question-region mapping for R1 student process reconstruction."""

from .mapping import DEFAULT_MIN_COVERAGE, map_strokes_to_regions
from .models import MappingStatus, StrokeMapping

__all__ = [
    "DEFAULT_MIN_COVERAGE",
    "MappingStatus",
    "StrokeMapping",
    "map_strokes_to_regions",
]
