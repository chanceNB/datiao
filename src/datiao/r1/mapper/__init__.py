"""Question-region mapping for R1 student process reconstruction."""

from .mapping import DEFAULT_AMBIGUITY_EPSILON, DEFAULT_MIN_COVERAGE, MAPPING_ALGORITHM_VERSION, QuestionMappingConfig, map_strokes_to_regions
from .models import MappingMethod, MappingStatus, StrokeMapping

__all__ = [
    "DEFAULT_MIN_COVERAGE",
    "DEFAULT_AMBIGUITY_EPSILON",
    "MAPPING_ALGORITHM_VERSION",
    "QuestionMappingConfig",
    "MappingMethod",
    "MappingStatus",
    "StrokeMapping",
    "map_strokes_to_regions",
]
