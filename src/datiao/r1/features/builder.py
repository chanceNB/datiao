"""Formal R1 feature materialization CLI; consumes only a reloaded dataset."""
from __future__ import annotations
import argparse, json, math, shutil
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any
from ..dataset.io import reload_dataset
from ..mapper import StrokeMapping
from ..models import Point, Stroke
from ..synthetic.models import SyntheticTruthEvent
from .models import *
from .alignment import align_truth, target_vector
from .audit import run_alignment_audit, run_feature_leakage_audit, run_feature_range_audit, run_feature_split_audit
from .manifest import compute_feature_manifest_hash, sha256_file, write_json, write_jsonl

def load_feature_config(path: str|Path|None) -> FeatureConfig:
    if path is None: return FeatureConfig()
    return FeatureConfig.model_validate(json.loads(Path(path).read_text(encoding='utf-8')))

def _bbox(points, normalized=True):
    vals=[(p.x_norm,p.y_norm) if normalized else (p.x_raw,p.y_raw) for p in points]
    if not vals or any(x is None or y is None for x,y in vals): return None
    xs=[x for x,y in vals]; ys=[y for x,y in vals]
    return (min(xs),min(ys),max(xs),max(ys))
def _iou(a,b):
    if not a or not b:return None
    ix=max(0,min(a[2],b[2])-max(a[0],b[0])); iy=max(0,min(a[3],b[3])-max(a[1],b[1]))
    inter=ix*iy; ua=(a[2]-a[0])*(a[3]-a[1]); ub=(b[2]-b[0])*(b[3]-b[1]); union=ua+ub-inter
    return 0.0 if union<=0 else inter/union

def materialize_features(dataset: str|Path, output: str|Path, config: FeatureConfig|None=None, overwrite=False) -> FeatureBuildResult:
    cfg=config or FeatureConfig(); source=reload_dataset(dataset); root=Path(output)
    if root.exists() and any(root.iterdir()):
        if not overwrite: raise FileExistsError(f'feature output exists: {root}')
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    cases={c.case_id:c for c in source['cases']}; source_splits={c.case_id:c.split for c in source['cases']}
    points=defaultdict(dict); strokes=defaultdict(dict); maps=defaultdict(dict); truths=defaultdict(list)
    for p in source['points']: points[p.session_id][p.point_id]=p
    # records carry the case association; model collections are grouped by session and ids are unique per case.
    for rec in source['point_records']: points[rec['case_id']][rec['point']['point_id']]=Point.model_validate(rec['point'])
    for rec in source['stroke_records']: strokes[rec['case_id']][rec['stroke']['stroke_id']]=Stroke.model_validate(rec['stroke'])
    for rec in source['mapping_records']:
        payload=dict(rec['mapping']); payload['point_refs']=tuple(payload.get('point_refs',())); payload['quality_flags']=tuple(payload.get('quality_flags',()))
        maps[rec['case_id']][payload['stroke_id']]=StrokeMapping.model_validate(payload)
    for rec in source['truth_records']:
        payload=dict(rec['event']); payload['source_point_ids']=tuple(payload.get('source_point_ids',()))
        truths[rec['case_id']].append(SyntheticTruthEvent.model_validate(payload))
    episodes=[]; rows=[]; previous={}; question_seen=defaultdict(int); last_question={}; last_bbox={}
    for case_id in sorted(cases):
        case=cases[case_id]; ordered=sorted(strokes[case_id].values(), key=lambda s: (s.start_time_ms is None, s.start_time_ms or 0, s.stroke_id))
        prev_end=None
        for idx, stroke in enumerate(ordered):
            mapping=maps[case_id].get(stroke.stroke_id)
            status=mapping.status if mapping else 'UNKNOWN'; method=mapping.mapping_method if mapping else 'UNKNOWN'
            flags=set(stroke.quality_flags) | (set(mapping.quality_flags) if mapping else set())
            pts=[points[case_id][pid] for pid in stroke.processed_order if pid in points[case_id]]
            flags |= {f for p in pts for f in p.quality_flags}
            q=mapping.question_id if mapping else None; page=mapping.page_id if mapping and mapping.page_id else stroke.page_id
            key=(case.participant_id,case.task_segment_id,q)
            visit=None if q is None else question_seen[key]+1
            ret=None if q is None else (1 if last_question.get((case.participant_id,case.task_segment_id)) is not None and last_question[(case.participant_id,case.task_segment_id)] != q and question_seen[key]>0 else 0)
            if q is not None: question_seen[key]+=1; last_question[(case.participant_id,case.task_segment_id)]=q
            duration=None if stroke.start_time_ms is None or stroke.end_time_ms is None else (stroke.end_time_ms-stroke.start_time_ms)/1000
            pause_before=None if prev_end is None or stroke.start_time_ms is None else max(0,(stroke.start_time_ms-prev_end))/1000
            ts=[p.timestamp_ms for p in pts]; pause_inside=None
            if len(ts)>=2 and all(t is not None for t in ts): pause_inside=sum(max(0,ts[i]-ts[i-1]-cfg.inside_pause_threshold_ms) for i in range(1,len(ts)))/1000
            path=stroke.provenance.get('path_length_norm') if isinstance(stroke.provenance,dict) else None
            if not isinstance(path,(int,float)):
                path=sum(math.hypot(pts[i].x_norm-pts[i-1].x_norm,pts[i].y_norm-pts[i-1].y_norm) for i in range(1,len(pts))) if pts and all(p.x_norm is not None and p.y_norm is not None for p in pts) else None
            speed=None if path is None or duration is None or duration<=0 else path/duration
            bboxw=bboxh=None
            if isinstance(stroke.provenance,dict) and stroke.provenance.get('bbox_coordinate_space')=='norm': bboxw,bboxh=stroke.bbox.width,stroke.bbox.height
            occupancy=None if mapping is None or mapping.status!='MAPPED' else mapping.confidence
            geom=_bbox(pts, normalized=True); ink=None
            prior=last_bbox.get((case.participant_id,case.task_segment_id,q,page)) if q is not None else None
            if prior is not None and geom is not None: ink=_iou(geom,prior)
            if q is not None and geom is not None: last_bbox[(case.participant_id,case.task_segment_id,q,page)]=geom
            ep=ObservationEpisode(episode_id=f'{case_id}::{stroke.stroke_id}',case_id=case_id,dataset_id=case.dataset_id,dataset_version=case.dataset_version,participant_id=case.participant_id,session_id=case.session_id,task_segment_id=case.task_segment_id,device_id=case.device_id,split=case.split,stroke_id=stroke.stroke_id,point_refs=stroke.point_refs,page_id=page,question_id=q,mapping_status=status,mapping_method=method,start_time_ms=stroke.start_time_ms,end_time_ms=stroke.end_time_ms,episode_index=idx,question_visit_index=visit,question_return_count=ret,quality_status='VALID' if status=='MAPPED' and not flags else 'DEGRADED',quality_flags=tuple(sorted(flags)))
            feats={"duration_s":duration,"path_length_norm":path,"mean_speed_norm_per_s":speed,"pause_before_s":pause_before,"pause_inside_s":pause_inside,"bbox_width_norm":bboxw,"bbox_height_norm":bboxh,"question_occupancy":occupancy,"visit_index":visit,"return_count":ret,"previous_ink_iou":ink,"quality_valid":1.0 if ep.quality_status=='VALID' else 0.0}
            episodes.append(ep); rows.append((ep,feats)); prev_end=stroke.end_time_ms if stroke.end_time_ms is not None else prev_end
    labels, alignment=align_truth(episodes,truths)
    alignment_audit=run_alignment_audit(alignment)
    if alignment_audit['status']!='PASS': raise ValueError(f'truth alignment failed: {alignment_audit}')
    feature_rows=[FeatureRow(episode_id=ep.episode_id,split=ep.split,features=feats,target=target_vector(labels[ep.episode_id])) for ep,feats in rows]
    sequences=[]
    for case_id in sorted(cases):
        es=[e for e in episodes if e.case_id==case_id]; rs=[r for r in feature_rows if r.episode_id in {e.episode_id for e in es}]
        if not es: continue
        sequences.append(SequenceSample(sequence_id=f'{case_id}::{cases[case_id].participant_id}::{cases[case_id].task_segment_id}',case_id=case_id,participant_id=cases[case_id].participant_id,task_segment_id=cases[case_id].task_segment_id,split=es[0].split,episode_ids=tuple(e.episode_id for e in es),values=tuple(tuple(0.0 if v is None else float(v) for v in r.features.values()) for r in rs),feature_mask=tuple(tuple(0 if v is None else 1 for v in r.features.values()) for r in rs),targets=tuple(r.target for r in rs)))
    leakage=run_feature_leakage_audit(feature_rows); ranges=run_feature_range_audit(tuple(feature_rows)); splits=run_feature_split_audit(episodes,sequences,source_splits)
    if leakage['status']!='PASS' or ranges['status']!='PASS' or splits['status']!='PASS': raise ValueError({'leakage':leakage,'ranges':ranges,'splits':splits})
    for name, payload in [('episodes.jsonl',[e.model_dump(mode='json') for e in episodes]),('feature_rows.jsonl',[r.model_dump(mode='json') for r in feature_rows]),('sequence_samples.jsonl',[s.model_dump(mode='json') for s in sequences])]: write_jsonl(root/name,payload)
    split_files={}
    for split in ('train','validation','test'):
        epids=tuple(e.episode_id for e in episodes if e.split==split); seqids=tuple(s.sequence_id for s in sequences if s.split==split)
        split_files[split]={'split_name':split,'source_dataset_id':source['manifest'].dataset_id,'source_dataset_manifest_hash':source['manifest'].manifest_hash,'episode_ids':list(epids),'sequence_ids':list(seqids),'count':len(epids)}; write_json(root/'splits'/f'{split}.json',split_files[split])
    descriptions={k:("s","derived observable value","null when unavailable") for k in FEATURE_ORDER}
    schema={'schema_version':FEATURE_SCHEMA_VERSION,'feature_builder_version':FEATURE_BUILDER_VERSION,'ordered_features':[{'name':k,'dtype':'float|null' if k!='quality_valid' else 'float','unit':descriptions[k][0],'description':descriptions[k][1],'source':'reloaded Point/Stroke/Mapping observable history','missing_policy':descriptions[k][2],'model_input':True} for k in FEATURE_ORDER]}
    target={'target_spec_version':TARGET_SPEC_VERSION,'supervision_granularity':'episode','task_type':'multi_label_binary','label_order':list(LABEL_ORDER),'encoding':'multi_hot','label_source':'independent_synthetic_truth','alignment_version':TRUTH_ALIGNMENT_VERSION,'allowed_overlap':True,'ignore_policy':'unmatched or ambiguous truth fails build','unlabeled_episode_policy':'all_zero_valid_negative','future_lightgbm_strategy':'8 one-vs-rest binary classifiers','future_tcn_loss':'8 logits with BCEWithLogitsLoss'}
    write_json(root/'feature_schema.json',schema); write_json(root/'target_spec.json',target); write_json(root/'alignment_audit.json',alignment_audit); write_json(root/'feature_leakage_audit.json',leakage); write_json(root/'range_audit.json',ranges); write_json(root/'split_inheritance_audit.json',splits)
    miss={}
    for key in FEATURE_ORDER:
        missing=sum(r.features[key] is None for r in feature_rows); miss[key]={'overall':{'observed':len(feature_rows)-missing,'missing':missing,'missing_rate':missing/len(feature_rows)}}
        for split in ('train','validation','test'):
            sr=[r for r in feature_rows if r.split==split]; sm=sum(r.features[key] is None for r in sr); miss[key][split]={'observed':len(sr)-sm,'missing':sm,'missing_rate':sm/len(sr) if sr else 0.0}
    target_dist={label:sum(r.target[i] for r in feature_rows) for i,label in enumerate(LABEL_ORDER)}
    cooccurrence={f'{LABEL_ORDER[i]}+{LABEL_ORDER[j]}':sum(r.target[i] and r.target[j] for r in feature_rows) for i in range(len(LABEL_ORDER)) for j in range(i+1,len(LABEL_ORDER)) if any(r.target[i] and r.target[j] for r in feature_rows)}
    core=['episodes.jsonl','feature_rows.jsonl','sequence_samples.jsonl','feature_schema.json','target_spec.json']+[f'splits/{s}.json' for s in ('train','validation','test')]
    manifest=FeatureManifest(feature_dataset_id=cfg.feature_dataset_id,feature_dataset_version=cfg.feature_dataset_version,source_dataset_id=source['manifest'].dataset_id,source_dataset_version=source['manifest'].dataset_version,source_dataset_manifest_hash=source['manifest'].manifest_hash,source_split_manifest_hash=source['manifest'].split_manifest_hash,feature_builder_version=cfg.feature_builder_version,feature_schema_version=cfg.feature_schema_version,target_spec_version=cfg.target_spec_version,truth_alignment_version=cfg.truth_alignment_version,episode_count=len(episodes),sequence_count=len(sequences),feature_order=FEATURE_ORDER,label_order=LABEL_ORDER,split_counts={s:{'episodes':sum(e.split==s for e in episodes),'sequences':sum(x.split==s for x in sequences)} for s in ('train','validation','test')},missingness_summary=miss,target_distribution=target_dist,file_hashes={p:sha256_file(root/p) for p in core},files=tuple(core))
    manifest=manifest.model_copy(update={'manifest_hash':compute_feature_manifest_hash(manifest)})
    write_json(root/'feature_manifest.json',manifest.model_dump(mode='json')); summary={'manifest_hash':manifest.manifest_hash,'episode_count':len(episodes),'sequence_count':len(sequences),'source_dataset_manifest_hash':source['manifest'].manifest_hash,'alignment_audit':alignment_audit,'leakage_audit':leakage,'range_audit':ranges,'split_inheritance_audit':splits,'missingness_summary':miss,'target_distribution':target_dist,'target_cooccurrence':cooccurrence}
    write_json(root/'feature_summary.json',summary)
    return FeatureBuildResult(output_path=str(root),manifest=manifest,episode_count=len(episodes),sequence_count=len(sequences))

build_feature_dataset = materialize_features

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',required=True); ap.add_argument('--output',required=True); ap.add_argument('--config'); ap.add_argument('--overwrite',action='store_true'); args=ap.parse_args(argv)
    result=materialize_features(args.dataset,args.output,load_feature_config(args.config),args.overwrite); print(json.dumps(result.model_dump(mode='json'),ensure_ascii=False,indent=2))
if __name__=='__main__': main()
