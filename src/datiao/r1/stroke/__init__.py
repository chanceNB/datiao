"""Stroke construction for R1."""

from .builder import StrokeBuildConfig, build_strokes
from .geometry import DistanceResult, path_length, point_distance

__all__ = ["DistanceResult", "StrokeBuildConfig", "build_strokes", "path_length", "point_distance"]
