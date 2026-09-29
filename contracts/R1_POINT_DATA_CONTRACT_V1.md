# R1 Point Data Contract V1

**Version:** `1.0.0`  
**Schema dialect:** JSON Schema Draft 2020-12

## Purpose

This contract is the stable R1 boundary for downstream consumers. It serializes canonical points, strokes, normalized question regions and student process events while retaining immutable Raw provenance for audit and trace.

## Point

A Point separates raw device coordinates (`x_raw`, `y_raw`) from calibrated millimetres (`x_mm`, `y_mm`) and template normalized coordinates (`x_norm`, `y_norm`). Unknown calibration remains `null`. `participant_id` and `task_segment_id` remain nullable when the adapter cannot prove them. `pressure_raw=0` means a recorded zero and is distinct from `null`. `pen_state` is one of `DOWN`, `MOVE`, `UP`, `UNKNOWN`. `source_payload` and its SHA-256 hash remain internal provenance and are never overwritten by canonicalization.

The internal Point model retains degraded records for raw retention and quality audit. A successful Canonical Point V1 export must pass `is_canonical_v1_eligible()` (or `validate_canonical_point_for_export()`): `device_id`, `page_id`, `timestamp_ms`, `x_raw` and `y_raw` must all be present. A degraded Point is never labeled as a successful standard export and is never dropped merely because one of these values is missing.

## Stroke

A Stroke references Points through `point_refs`, keeps `raw_order` and `processed_order`, and uses `start_time_ms`, `end_time_ms`, `duration_ms`, `bbox`, `quality_flags`, `algorithm_version` and `provenance`. Advanced pen features are intentionally not part of this version.

## QuestionRegion

A standard region uses `region_type`, `polygon_norm` and `coordinate_space="norm"`. Normalized coordinates are between 0 and 1 and contain at least three vertices. Legacy rectangles use the separate `LegacyQuestionRegion` compatibility type and cannot validate against the V1 JSON Schema until an explicit normalization/calibration adapter produces a standard `QuestionRegion`.

## Student Event

The event enum is exactly:

`WRITING`, `QUESTION_VISIT`, `QUESTION_LEAVE`, `RETURN`, `REVISION_CANDIDATE`, `PAGE_CHANGE`, `PROCESS_END`, `UNKNOWN`.

Events use `start_time_ms`/`end_time_ms`, `point_refs`/`stroke_refs`, `quality_status`, `quality_flags`, `algorithm_version` and `provenance`. `end_time_ms >= start_time_ms` whenever both are present. `VALID` events require both times. A degraded or invalid event may omit a time only with an explanatory flag such as `TIME_UNAVAILABLE`. `QUESTION_VISIT`, `QUESTION_LEAVE`, `RETURN` and `REVISION_CANDIDATE` require `question_id`; mapping failure emits an `UNKNOWN` event with `question_id="UNKNOWN"`. `PAGE_CHANGE` may leave it null. `quality_status` is the total status (`VALID`, `DEGRADED`, `INVALID`); `quality_flags` are reason codes. `UNKNOWN` is an observable event type and is not equivalent to invalid data.

Synthetic events with `dataset_type=synthetic` require `dataset_type`, `generator_version`, integer `seed`, `scenario_id`, `ground_truth_source` and a real `sha256:<64 hex>` `manifest_hash`. The hash is computed from a separate frozen Synthetic Manifest payload; the Manifest also keeps an independent `raw_records_hash`. The hash never includes a `manifest_hash` field in its own input.

## Breaking migration and compatibility

V1 standard objects do not serialize legacy fields such as `occurred_at_ms`, `source_point_ids` or `source_stroke_ids`. Legacy payloads must go through named helpers in `datiao.r1.models.legacy`; V1 models reject legacy constructor fields. Read-only aliases remain only for internal trace compatibility. Device aliases still require explicit adapters. No adapter guesses physical calibration or device-specific time semantics.

## Feature boundary

Provenance, scenario and truth metadata are audit data. A future Feature Builder must use a whitelist and must not flatten `model_dump()` wholesale into training features. R1-05 owns algorithm semantic upgrades such as arc-length mapping, IoU revision and timeout process-end.

