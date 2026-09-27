"""Deterministic synthetic scenarios with independent truth and real outputs."""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..event import detect_student_process_events
from ..mapper import map_strokes_to_regions
from ..models import Point, QuestionRegion
from ..parser import parse_raw_points
from ..stroke import build_strokes
from .models import (
    Scenario,
    SyntheticAlgorithmOutput,
    SyntheticCase,
    SyntheticTruth,
    SyntheticTruthEvent,
)


@dataclass(frozen=True, slots=True)
class _Action:
    page_id: str
    question_id: str | None
    x: float
    y: float
    expected_unknown: bool = False
    missing_timestamp: bool = False
    duplicate_point: bool = False
    out_of_order: bool = False


def _action_plan(scenario: Scenario) -> tuple[_Action, ...]:
    q1 = _Action("page-01", "q-01", 5.0, 5.0)
    q2 = _Action("page-01", "q-02", 25.0, 5.0)
    q3 = _Action("page-01", "q-03", 45.0, 5.0)
    page2 = _Action("page-02", "q-04", 5.0, 5.0)
    plans = {
        "sequential_visit": (q1, q2, q3),
        "return_visit": (q1, q2, q1),
        "revision_candidate": (q1, q1),
        "cross_question_jump": (q1, q3),
        "page_change": (q1, page2),
        "pause_resume": (q1, q1),
        "unknown_region": (
            q1,
            _Action("page-01", None, 100.0, 100.0, expected_unknown=True),
        ),
        "missing_point": (
            q1,
            _Action("page-01", "q-02", 25.0, 5.0, missing_timestamp=True),
        ),
        "duplicate_point": (
            q1,
            _Action("page-01", "q-02", 25.0, 5.0, duplicate_point=True),
        ),
        "out_of_order": (
            q1,
            _Action("page-01", "q-02", 25.0, 5.0, out_of_order=True),
        ),
    }
    return plans[scenario.scenario_type]


def default_regions() -> tuple[QuestionRegion, ...]:
    return (
        QuestionRegion(
            region_id="region-q-01",
            page_id="page-01",
            question_id="q-01",
            geometry_type="rectangle",
            coordinates=(0.0, 0.0, 10.0, 10.0),
        ),
        QuestionRegion(
            region_id="region-q-02",
            page_id="page-01",
            question_id="q-02",
            geometry_type="rectangle",
            coordinates=(20.0, 0.0, 10.0, 10.0),
        ),
        QuestionRegion(
            region_id="region-q-03",
            page_id="page-01",
            question_id="q-03",
            geometry_type="rectangle",
            coordinates=(40.0, 0.0, 10.0, 10.0),
        ),
        QuestionRegion(
            region_id="region-q-04",
            page_id="page-02",
            question_id="q-04",
            geometry_type="rectangle",
            coordinates=(0.0, 0.0, 10.0, 10.0),
        ),
    )


def _raw_records(scenario: Scenario) -> tuple[dict[str, object], ...]:
    rng = random.Random(scenario.seed)
    records: list[dict[str, object]] = []
    point_counter = 0
    for action_index, action in enumerate(_action_plan(scenario)):
        for point_index in range(2):
            point_id = f"{scenario.scenario_id}-p-{point_counter:03d}"
            if action.duplicate_point and point_index == 1:
                point_id = f"{scenario.scenario_id}-p-{point_counter - 1:03d}"
            timestamp: int | None = 1_000 + action_index * 2_000 + point_index * 50
            if action.missing_timestamp and point_index == 0:
                timestamp = None
            sequence = action_index * 2 + point_index
            if action.out_of_order and point_index == 1:
                sequence -= 2
            records.append(
                {
                    "point_id": point_id,
                    "session_id": f"session-{scenario.scenario_id}",
                    "page_id": action.page_id,
                    "x": action.x + rng.uniform(-0.5, 0.5),
                    "y": action.y + rng.uniform(-0.5, 0.5),
                    "timestamp_ms": timestamp,
                    "sequence": sequence,
                    "scenario_id": scenario.scenario_id,
                    "question_id": action.question_id,
                    "generator_version": scenario.generator_version,
                }
            )
            point_counter += 1
    return tuple(records)


def generate_raw_points(scenario: Scenario) -> tuple[Point, ...]:
    """Generate canonical points from a deterministic scenario plan."""

    return parse_raw_points(_raw_records(scenario))


def _truth_for(scenario: Scenario, raw_points: tuple[Point, ...]) -> SyntheticTruth:
    """Build truth from the scenario plan, independently of the detector."""

    actions = _action_plan(scenario)
    truth_events: list[SyntheticTruthEvent] = []
    visited: set[str] = set()
    active_question: str | None = None
    active_page: str | None = None

    def add(event_type: str, action_index: int, question_id: str | None = None) -> None:
        start = action_index * 2
        source_ids = tuple(
            point.point_id for point in raw_points[start : min(start + 2, len(raw_points))]
        )
        truth_events.append(
            SyntheticTruthEvent(
                event_id=f"{scenario.scenario_id}-truth-{len(truth_events):04d}",
                event_type=event_type,
                page_id=actions[action_index].page_id,
                question_id=question_id,
                source_point_ids=source_ids,
            )
        )

    for index, action in enumerate(actions):
        if active_page is not None and action.page_id != active_page:
            add("PAGE_CHANGE", index)
        active_page = action.page_id

        if action.expected_unknown:
            if active_question is not None:
                add("QUESTION_LEAVE", index - 1, active_question)
                active_question = None
            add("UNKNOWN", index)
            continue

        question_id = action.question_id
        assert question_id is not None
        if active_question is None:
            add("RETURN" if question_id in visited else "QUESTION_VISIT", index, question_id)
        elif active_question == question_id:
            add("REVISION_CANDIDATE", index, question_id)
        else:
            add("QUESTION_LEAVE", index - 1, active_question)
            add("RETURN" if question_id in visited else "QUESTION_VISIT", index, question_id)
        visited.add(question_id)
        active_question = question_id

    if actions:
        add("PROCESS_END", len(actions) - 1, active_question)
    return SyntheticTruth(scenario_id=scenario.scenario_id, truth_events=tuple(truth_events))


def generate_synthetic_case(scenario: Scenario) -> SyntheticCase:
    raw_points = generate_raw_points(scenario)
    regions = default_regions()
    strokes = build_strokes(raw_points)
    mappings = map_strokes_to_regions(strokes, raw_points, regions)
    events = detect_student_process_events(mappings)
    truth = _truth_for(scenario, raw_points)
    algorithm_output = SyntheticAlgorithmOutput(
        algorithm_version="r1.algorithm.v1",
        implemented=True,
        events=events,
    )
    return SyntheticCase(
        scenario=scenario,
        raw_points=raw_points,
        regions=regions,
        strokes=strokes,
        mappings=mappings,
        truth=truth,
        algorithm_output=algorithm_output,
    )
