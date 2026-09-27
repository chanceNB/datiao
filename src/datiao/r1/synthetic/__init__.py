"""Reproducible R1 synthetic input and truth framework."""

from .generator import default_regions, generate_raw_points, generate_synthetic_case
from .models import (
    Scenario,
    SyntheticAlgorithmOutput,
    SyntheticCase,
    SyntheticTruth,
    SyntheticTruthEvent,
)

__all__ = [
    "Scenario",
    "SyntheticAlgorithmOutput",
    "SyntheticCase",
    "SyntheticTruth",
    "SyntheticTruthEvent",
    "default_regions",
    "generate_raw_points",
    "generate_synthetic_case",
]
