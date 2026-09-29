from __future__ import annotations
from pathlib import Path
from .manifest import read_json, read_jsonl, sha256_file, compute_feature_manifest_hash
from .models import FeatureManifest, FeatureRow, ObservationEpisode, SequenceSample, FeatureSplitManifest, FEATURE_ORDER, LABEL_ORDER
from .audit import run_feature_leakage_audit, run_feature_range_audit, run_feature_split_audit

class ReloadedFeatureDataset(dict):
    pass

def reload_feature_dataset(output: str|Path) -> ReloadedFeatureDataset:
    root=Path(output); manifest=FeatureManifest.model_validate(read_json(root/'feature_manifest.json'))
    if manifest.manifest_hash != compute_feature_manifest_hash(manifest): raise ValueError('feature manifest hash mismatch')
    if tuple(manifest.feature_order)!=FEATURE_ORDER or tuple(manifest.label_order)!=LABEL_ORDER: raise ValueError('frozen order mismatch')
    for rel, expected in manifest.file_hashes.items():
        if sha256_file(root/rel)!=expected: raise ValueError(f'file hash mismatch: {rel}')
    episodes=tuple(ObservationEpisode.model_validate(x) for x in read_jsonl(root/'episodes.jsonl'))
    rows=tuple(FeatureRow.model_validate(x) for x in read_jsonl(root/'feature_rows.jsonl'))
    sequences=tuple(SequenceSample.model_validate(x) for x in read_jsonl(root/'sequence_samples.jsonl'))
    if len(episodes)!=manifest.episode_count or len(sequences)!=manifest.sequence_count or len(rows)!=len(episodes): raise ValueError('count mismatch')
    if {e.episode_id for e in episodes}!={r.episode_id for r in rows}: raise ValueError('episode/row ids mismatch')
    source_splits={e.case_id:e.split for e in episodes}
    leakage=run_feature_leakage_audit(rows); ranges=run_feature_range_audit(rows); split=run_feature_split_audit(episodes,sequences,source_splits)
    if any(a['status']!='PASS' for a in (leakage,ranges,split)): raise ValueError('recomputed feature audit failed')
    stored=read_json(root/'feature_leakage_audit.json')
    if stored!=leakage: raise ValueError('stored leakage audit differs')
    stored=read_json(root/'range_audit.json')
    if stored!=ranges: raise ValueError('stored range audit differs')
    stored=read_json(root/'split_inheritance_audit.json')
    if stored!=split: raise ValueError('stored split audit differs')
    summary=read_json(root/'feature_summary.json')
    if summary.get('manifest_hash')!=manifest.manifest_hash: raise ValueError('summary hash mismatch')
    return ReloadedFeatureDataset(output_path=str(root),manifest=manifest,episodes=episodes,feature_rows=rows,sequences=sequences,leakage_audit=leakage,range_audit=ranges,split_audit=split,summary=summary)
