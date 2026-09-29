from datiao.r1.synthetic import Scenario, generate_synthetic_case


def test_synthetic_v1_ids_time_and_provenance_are_explicit():
    case = generate_synthetic_case(Scenario("return_q5_001", "return_visit", seed=20260929))
    assert case.raw_points[0].session_id == "sim_session_return_q5_001"
    assert case.raw_points[0].participant_id == "sim_p_001"
    assert case.raw_points[0].task_segment_id == "sim_segment_practice_01"
    assert min(point.timestamp_ms for point in case.raw_points if point.timestamp_ms is not None) == 0
    assert case.algorithm_output.events
    provenance = case.algorithm_output.events[0].provenance
    assert set(("dataset_type", "generator_version", "seed", "scenario_id", "ground_truth_source", "manifest_hash")) <= set(provenance)
    assert provenance["dataset_type"] == "synthetic"
    assert isinstance(provenance["seed"], int)
    assert provenance["manifest_hash"].startswith("sha256:")
    assert "REPLACE_WITH_REAL_HASH" not in provenance["manifest_hash"]


def test_synthetic_truth_is_independent_from_prediction_ids():
    case = generate_synthetic_case(Scenario("return_q5_002", "return_visit", seed=7))
    assert case.truth is not case.algorithm_output
    assert case.truth.truth_events is not case.algorithm_output.events
    assert all(event.event_id != predicted.event_id for event in case.truth.truth_events for predicted in case.algorithm_output.events)
