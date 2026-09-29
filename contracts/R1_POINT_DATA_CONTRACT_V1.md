# R1 Point Data Contract V1

**Version:** `1.0.0`  
**Schema dialect:** JSON Schema Draft 2020-12

## Purpose

This contract is the stable R1 boundary for downstream consumers. It serializes canonical points, strokes, normalized question regions and student process events while retaining immutable Raw provenance for audit and trace.

## Point

A Point separates raw device coordinates (`x_raw`, `y_raw`) from calibrated millimetres (`x_mm`, `y_mm`) and template normalized coordinates (`x_norm`, `y_norm`). Unknown calibration remains `null`. `participant_id` and `task_segment_id` remain nullable when the adapter cannot prove them. `pressure_raw=0` means a recorded zero and is distinct from `null`. `pen_state` is one of `DOWN`, `MOVE`, `UP`, `UNKNOWN`. `source_payload` and its SHA-256 hash remain internal provenance and are never overwritten by canonicalization.

## Stroke

A Stroke references Points through `point_refs`, keeps `raw_order` and `processed_order`, and uses `start_time_ms`, `end_time_ms`, `duration_ms`, `bbox`, `quality_flags`, `algorithm_version` and `provenance`. Advanced pen features are intentionally not part of this version.

## QuestionRegion

A region uses `region_type` and `polygon_norm`. Normalized coordinates are between 0 and 1 and contain at least three vertices. Legacy rectangle inputs are accepted only through the compatibility constructor and retain `coordinate_space=legacy`; they are not silently relabeled as normalized.

## Student Event

The event enum is exactly:

`WRITING`, `QUESTION_VISIT`, `QUESTION_LEAVE`, `RETURN`, `REVISION_CANDIDATE`, `PAGE_CHANGE`, `PROCESS_END`, `UNKNOWN`.

Events use `start_time_ms`/`end_time_ms`, `point_refs`/`stroke_refs`, `quality_status`, `quality_flags`, `algorithm_version` and `provenance`. `end_time_ms >= start_time_ms` whenever both are present. `quality_status` is the total status (`VALID`, `DEGRADED`, `INVALID`); `quality_flags` are reason codes. `UNKNOWN` is an observable event type and is not equivalent to invalid data.

Synthetic events with `dataset_type=synthetic` require `dataset_type`, `generator_version`, integer `seed`, `scenario_id`, `ground_truth_source` and a real `sha256:<64 hex>` `manifest_hash`.

## Breaking migration and compatibility

V1 standard objects do not serialize legacy fields such as `occurred_at_ms`, `source_point_ids` or `source_stroke_ids`. Legacy payloads must go through named helpers in `datiao.r1.models.legacy`; V1 models reject legacy constructor fields. Read-only aliases remain only for internal trace compatibility. Device aliases still require explicit adapters. No adapter guesses physical calibration or device-specific time semantics.

## Feature boundary

Provenance, scenario and truth metadata are audit data. A future Feature Builder must use a whitelist and must not flatten `model_dump()` wholesale into training features. R1-05 owns algorithm semantic upgrades such as arc-length mapping, IoU revision and timeout process-end.

