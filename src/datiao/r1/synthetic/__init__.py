"""Reproducible R1 synthetic input and truth framework."""

from .generator import default_regions, generate_raw_points, generate_synthetic_case
from .manifest import SyntheticManifest, build_synthetic_manifest, compute_manifest_hash
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
    "SyntheticManifest",
    "build_synthetic_manifest",
    "compute_manifest_hash",
    "default_regions",
    "generate_raw_points",
    "generate_synthetic_case",
]
