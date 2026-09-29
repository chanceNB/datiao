from datiao.r1.synthetic import Scenario, generate_synthetic_case


def test_same_seed_generates_same_case():
    first = generate_synthetic_case(Scenario("case-001", "sequential_visit", seed=7))
    second = generate_synthetic_case(Scenario("case-001", "sequential_visit", seed=7))

    assert first == second
    assert first.raw_points == second.raw_points


def test_supported_scenarios_generate_raw_points_and_truth():
    for scenario_type in (
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
        "continuous_same_question_writing",
        "true_revision_overlap",
        "same_question_no_overlap",
        "explicit_process_end",
        "open_process_no_end",
        "spatial_jump_split",
        "arc_length_cross_region",
    ):
        case = generate_synthetic_case(Scenario(f"case-{scenario_type}", scenario_type, seed=1))

        assert case.raw_points
        assert case.truth.truth_events
        assert case.algorithm_output.implemented is True
        assert case.algorithm_output.events
        assert case.truth.scenario_id == case.scenario.scenario_id


def test_truth_and_algorithm_output_are_separate():
    case = generate_synthetic_case(Scenario("case-002", "return_visit", seed=2))

    assert case.truth.truth_events
    assert case.algorithm_output.events
    assert case.algorithm_output.implemented is True
    assert case.truth.truth_events is not case.algorithm_output.events
    assert [event.event_type for event in case.truth.truth_events] == [
        event.event_type for event in case.algorithm_output.events
    ]


def test_synthetic_truth_is_not_an_algorithm_event_type():
    case = generate_synthetic_case(Scenario("case-003", "revision_candidate", seed=3))

    assert all(hasattr(event, "event_type") for event in case.truth.truth_events)
    assert all(hasattr(event, "event_type") for event in case.algorithm_output.events)


def test_quality_scenarios_degrade_to_unknown_without_fake_success():
    for scenario_type in ("duplicate_point", "out_of_order"):
        case = generate_synthetic_case(Scenario(f"case-{scenario_type}", scenario_type, seed=1))

        assert any(mapping.status == "UNKNOWN" for mapping in case.mappings)
        assert any(event.event_type == "UNKNOWN" for event in case.algorithm_output.events)
