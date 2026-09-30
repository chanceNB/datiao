# R1 ↔ R3 Integration Checklist

- [x] Event enum is exactly WRITING, QUESTION_VISIT, QUESTION_LEAVE, RETURN, REVISION_CANDIDATE, PAGE_CHANGE, PROCESS_END, UNKNOWN.
- [x] Synthetic `session_id` and `participant_id` use the `sim_` prefix.
- [x] `task_segment_id` is copied or `null`; it is never guessed.
- [x] Unmapped `question_id` projects to `"UNKNOWN"`.
- [x] Actual detector algorithm version is preserved.
- [x] Dataset version is explicitly supplied to the batch exporter.
- [x] Synthetic provenance carries generator, seed, scenario, ground truth source, and real manifest hash.
- [x] Point and stroke refs are resolved before export.
- [x] VALID, DEGRADED, and INVALID quality statuses are preserved.
- [x] Batch session, task segment, and algorithm versions are consistent.
- [x] Strict Draft 2020-12 schema validation passes for the golden event.
- [x] R1 Core Event and R3 transport payload remain separate.
- [ ] Live R1↔R3 integration acceptance by R3.
