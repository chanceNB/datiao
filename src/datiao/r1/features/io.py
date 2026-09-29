from __future__ import annotations
from pathlib import Path
from .manifest import read_json, read_jsonl, sha256_file, compute_feature_manifest_hash
from .models import AlignmentRecord, FeatureManifest, FeatureRow, ObservationEpisode, SequenceSample, FeatureSplitManifest, FEATURE_ORDER, LABEL_ORDER
from .audit import run_alignment_audit, run_feature_leakage_audit, run_feature_range_audit, run_feature_split_audit

class ReloadedFeatureDataset(dict):
    pass

def reload_feature_dataset(output: str|Path) -> ReloadedFeatureDataset:
    root=Path(output); manifest=FeatureManifest.model_validate(read_json(root/'feature_manifest.json'))
    if manifest.manifest_hash != compute_feature_manifest_hash(manifest): raise ValueError('feature manifest hash mismatch')
    if tuple(manifest.feature_order)!=FEATURE_ORDER or tuple(manifest.label_order)!=LABEL_ORDER: raise ValueError('frozen order mismatch')
    for rel, expected in manifest.file_hashes.items():
        if sha256_file(root/rel)!=expected: raise ValueError(f'file hash mismatch: {rel}')
    canonical_root = Path(__file__).resolve().parents[4]
    canonical_feature = read_json(canonical_root/'contracts'/'r1_feature_schema_v1.json')
    canonical_target = read_json(canonical_root/'contracts'/'r1_target_spec_v1.json')
    if read_json(root/'feature_schema.json') != canonical_feature or read_json(root/'target_spec.json') != canonical_target:
        raise ValueError('runtime contracts differ from canonical contracts')
    if manifest.feature_schema_version != canonical_feature['schema_version'] or manifest.feature_builder_version != canonical_feature['feature_builder_version']:
        raise ValueError('feature manifest schema lineage mismatch')
    if manifest.target_spec_version != canonical_target['target_spec_version'] or manifest.truth_alignment_version != canonical_target['alignment_version']:
        raise ValueError('feature manifest target lineage mismatch')
    episodes=tuple(ObservationEpisode.model_validate(x) for x in read_jsonl(root/'episodes.jsonl'))
    rows=tuple(FeatureRow.model_validate(x) for x in read_jsonl(root/'feature_rows.jsonl'))
    sequences=tuple(SequenceSample.model_validate(x) for x in read_jsonl(root/'sequence_samples.jsonl'))
    alignment_records=tuple(AlignmentRecord.model_validate(x) for x in read_jsonl(root/'alignment_records.jsonl'))
    if len(episodes)!=manifest.episode_count or len(sequences)!=manifest.sequence_count or len(rows)!=len(episodes): raise ValueError('count mismatch')
    if {e.episode_id for e in episodes}!={r.episode_id for r in rows}: raise ValueError('episode/row ids mismatch')
    split_manifests=tuple(FeatureSplitManifest.model_validate(read_json(root/'splits'/f'{name}.json')) for name in ('train','validation','test'))
    if any(sm.source_dataset_id != manifest.source_dataset_id or sm.source_dataset_manifest_hash != manifest.source_dataset_manifest_hash for sm in split_manifests):
        raise ValueError('feature split lineage mismatch')
    split_names={sm.split_name for sm in split_manifests}
    if split_names != {'train','validation','test'}: raise ValueError('feature split names incomplete')
    episode_ids={e.episode_id for e in episodes}; sequence_ids={s.sequence_id for s in sequences}
    split_episode_sets=[set(sm.episode_ids) for sm in split_manifests]; split_sequence_sets=[set(sm.sequence_ids) for sm in split_manifests]
    if set.union(*split_episode_sets) != episode_ids or sum(len(s) for s in split_episode_sets) != len(episode_ids): raise ValueError('feature episode split union/disjoint failure')
    if set.union(*split_sequence_sets) != sequence_ids or sum(len(s) for s in split_sequence_sets) != len(sequence_ids): raise ValueError('feature sequence split union/disjoint failure')
    for sm in split_manifests:
        if sm.count != len(sm.episode_ids) or sm.sequence_count != len(sm.sequence_ids): raise ValueError('split count mismatch')
        if any(next(e for e in episodes if e.episode_id==eid).split != sm.split_name for eid in sm.episode_ids): raise ValueError('episode split label mismatch')
        if any(next(s for s in sequences if s.sequence_id==sid).split != sm.split_name for sid in sm.sequence_ids): raise ValueError('sequence split label mismatch')
    source_splits={e.case_id:e.split for e in episodes}
    leakage=run_feature_leakage_audit(rows); ranges=run_feature_range_audit(rows); split=run_feature_split_audit(episodes,sequences,source_splits)
    alignment=run_alignment_audit(alignment_records)
    if any(a['status']!='PASS' for a in (leakage,ranges,split,alignment)): raise ValueError('recomputed feature audit failed')
    if any(eid not in episode_ids for record in alignment_records for eid in record.matched_episode_ids): raise ValueError('alignment references unknown episode')
    stored=read_json(root/'feature_leakage_audit.json')
    if stored!=leakage: raise ValueError('stored leakage audit differs')
    stored=read_json(root/'range_audit.json')
    if stored!=ranges: raise ValueError('stored range audit differs')
    stored=read_json(root/'split_inheritance_audit.json')
    if stored!=split: raise ValueError('stored split audit differs')
    if read_json(root/'alignment_audit.json') != alignment: raise ValueError('stored alignment audit differs')
    summary=read_json(root/'feature_summary.json')
    if summary.get('manifest_hash')!=manifest.manifest_hash: raise ValueError('summary hash mismatch')
    return ReloadedFeatureDataset(output_path=str(root),manifest=manifest,episodes=episodes,feature_rows=rows,sequences=sequences,alignment_records=alignment_records,split_manifests=split_manifests,leakage_audit=leakage,range_audit=ranges,split_audit=split,alignment_audit=alignment,summary=summary)
