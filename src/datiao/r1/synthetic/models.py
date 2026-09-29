"""Synthetic scenario, truth, and algorithm-output contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..mapper import StrokeMapping
from ..models import EventType, Point, QuestionRegion, Stroke, StudentProcessEvent
from .manifest import SyntheticManifest

ScenarioType = Literal[
    "sequential_visit",
    "return_visit",
    "revision_candidate",
    "cross_question_jump",
    "page_change",
    "unknown_page",
    "pause_resume",
    "unknown_region",
    "missing_point",
    "duplicate_point",
    "out_of_order",
]


@dataclass(frozen=True, slots=True)
class Scenario:
    scenario_id: str
    scenario_type: ScenarioType
    seed: int
    generator_version: str = "r1.synthetic.v1"

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError("scenario_id must not be empty")
        if self.scenario_type not in {
            "sequential_visit",
            "return_visit",
            "revision_candidate",
            "cross_question_jump",
            "page_change",
            "unknown_page",
            "pause_resume",
            "unknown_region",
            "missing_point",
            "duplicate_point",
            "out_of_order",
        }:
            raise ValueError(f"unsupported scenario_type: {self.scenario_type}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if not self.generator_version:
            raise ValueError("generator_version must not be empty")


class SyntheticTruthEvent(BaseModel):
    """Independent expected event description, never produced by the algorithm."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    event_id: str = Field(min_length=1)
    event_type: EventType
    page_id: str | None = None
    question_id: str | None = None
    source_point_ids: tuple[str, ...] = ()


class SyntheticTruth(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    scenario_id: str = Field(min_length=1)
    truth_events: tuple[SyntheticTruthEvent, ...]


class SyntheticAlgorithmOutput(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    algorithm_version: str = Field(min_length=1)
    implemented: bool
    events: tuple[StudentProcessEvent, ...] = ()


class SyntheticCase(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, strict=True, frozen=True, extra="forbid")

    scenario: Scenario
    manifest: SyntheticManifest
    raw_points: tuple[Point, ...]
    regions: tuple[QuestionRegion, ...] = ()
    strokes: tuple[Stroke, ...] = ()
    mappings: tuple[StrokeMapping, ...] = ()
    truth: SyntheticTruth
    algorithm_output: SyntheticAlgorithmOutput
