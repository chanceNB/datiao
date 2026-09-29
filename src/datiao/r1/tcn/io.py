"""TCN orchestration, LightGBM comparison, semantic manifests and reload."""
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from ..lightgbm.io import reload_lightgbm_run
from ..lightgbm.manifest import sha256_file, write_json
from ..features.models import LABEL_ORDER
from .data import load_tcn_dataset, Normalizer, build_sequence_length_audit, target_causality_audit, make_loader
from .manifest import semantic_json_hash, semantic_run_hash
from .model import PenTCN, model_state_hash
from .train import execute_training_once, evaluate


def _read_predictions(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line]


def _aggregate_from_records(records, labels):
    from ..evaluation.multilabel import calculate_subset_metrics
    y=np.asarray([r['target'] for r in records],dtype=np.int8); p=np.asarray([r['probability_vector'] for r in records],dtype=float)
    return calculate_subset_metrics(y,p,.5,labels)


def _per_label_deltas(tcn_metrics, lightgbm_metrics):
    """Compare the canonical eight-label metrics, including non-causal labels."""
    return {
        label: {
            'f1_delta': tcn_metrics['per_label'][label]['f1'] - lightgbm_metrics['per_label'][label]['f1'],
            'roc_auc_delta': None if (
                tcn_metrics['per_label'][label]['roc_auc'] is None
                or lightgbm_metrics['per_label'][label]['roc_auc'] is None
            ) else tcn_metrics['per_label'][label]['roc_auc'] - lightgbm_metrics['per_label'][label]['roc_auc'],
        }
        for label in LABEL_ORDER
    }


def compare_with_lightgbm(tcn_result, lightgbm_run):
    lgb=reload_lightgbm_run(lightgbm_run, tcn_result['data'].source_path)
    lgb_rows=_read_predictions(Path(lightgbm_run)/'predictions.jsonl'); tcn_rows=tcn_result['predictions']
    payload={'tcn_baseline_id':'r1-pen-tcn-v1','tcn_baseline_version':'1.0.0','tcn_model_state_hash':tcn_result['model_state_hash'],'lightgbm_run_hash':lgb['manifest']['run_hash'],'source_feature_manifest_hash':tcn_result['data'].manifest.manifest_hash}
    for split in ('validation','test'):
        a={r['episode_id']:r for r in tcn_rows if r['split']==split}; b={r['episode_id']:r for r in lgb_rows if r['split']==split}
        if set(a)!=set(b): raise ValueError(f'{split} episode ID mismatch')
        if any(a[e]['target'] != b[e]['target'] for e in a): raise ValueError(f'{split} target mismatch')
        payload[f'{split}_episode_set_match']=True; payload[f'{split}_target_match']=True
        payload.setdefault('canonical',{})[split]=tcn_result['metrics'][split]
        payload.setdefault('lightgbm_canonical',{})[split]=lgb['metrics'][split]
        causal_tcn=_aggregate_from_records(list(a.values()),('WRITING','QUESTION_VISIT','RETURN','REVISION_CANDIDATE','PAGE_CHANGE','UNKNOWN'))
        causal_lgb=_aggregate_from_records(list(b.values()),('WRITING','QUESTION_VISIT','RETURN','REVISION_CANDIDATE','PAGE_CHANGE','UNKNOWN'))
        payload.setdefault('causal_diagnostics',{})[split]=causal_tcn; payload.setdefault('lightgbm_causal_diagnostics',{})[split]=causal_lgb
        payload.setdefault('per_label_deltas',{})[split]=_per_label_deltas(
            tcn_result['metrics'][split], lgb['metrics'][split]
        )
    payload['label_order_match']=list(LABEL_ORDER)==lgb['manifest']['label_order'];
    if not payload['label_order_match']: raise ValueError('label order mismatch')
    return payload


def _write_manifest(result, comparison):
    root=Path(result['output_path']); comparison_hash=semantic_json_hash(comparison)
    artifact_files=('model/best_model.pt','normalization.json','sequence_length_audit.json','target_causality_audit.json','training_history.json','metrics.json','causal_diagnostics.json','predictions.jsonl','config_snapshot.json','environment.json','comparison_lightgbm.json','audits/input_integrity.json','audits/normalizer.json','audits/causality.json','audits/padding.json')
    artifact_hashes={relative:sha256_file(root/relative) for relative in artifact_files}
    model_file_hash=artifact_hashes['model/best_model.pt']; predictions_hash=artifact_hashes['predictions.jsonl']; metrics_hash=artifact_hashes['metrics.json']; diagnostics_hash=artifact_hashes['causal_diagnostics.json']
    payload={'run_version':'1.0.0','baseline_id':'r1-pen-tcn-v1','baseline_version':'1.0.0','source_feature_manifest_hash':result['data'].manifest.manifest_hash,'source_dataset_manifest_hash':result['data'].manifest.source_dataset_manifest_hash,'source_split_manifest_hash':result['data'].manifest.source_split_manifest_hash,'feature_order':list(result['data'].manifest.feature_order),'label_order':list(LABEL_ORDER),'sequence_count':len(result['data'].sequences),'real_timestep_count':len(result['predictions']),'split_counts':{s:sum(r['split']==s for r in result['predictions']) for s in ('train','validation','test')},'normalization_hash':result['normalizer'].semantic_hash,'config_snapshot_hash':artifact_hashes['config_snapshot.json'],'artifact_hashes':artifact_hashes,'architecture':{'input_channels':24,'hidden_channels':[32,32],'kernel_size':2,'dilations':[1,2],'convs_per_block':2,'dropout':.1,'output_channels':8,'receptive_field':result['model'].receptive_field,'parameter_count':result['model'].parameter_count},'seed':20260929,'device':'cpu','threshold':.5,'best_epoch':result['training'].best_epoch,'best_validation_loss':result['training'].best_validation_loss,'model_file_hash':model_file_hash,'model_state_hash':result['model_state_hash'],'predictions_hash':predictions_hash,'metrics_hash':metrics_hash,'causal_diagnostics_hash':diagnostics_hash,'comparison_hash':comparison_hash,'comparison':comparison}
    payload['run_hash']=semantic_run_hash(payload); write_json(root/'run_manifest.json',payload); return payload


def reload_tcn_run(output, features):
    root=Path(output); manifest=json.loads((root/'run_manifest.json').read_text(encoding='utf-8')); data=load_tcn_dataset(features)
    if manifest['source_feature_manifest_hash'] != data.manifest.manifest_hash: raise ValueError('source feature lineage mismatch')
    if sha256_file(root/'model'/'best_model.pt') != manifest['model_file_hash']: raise ValueError('model file hash mismatch')
    for relative, digest in manifest.get('artifact_hashes', {}).items():
        if sha256_file(root/relative) != digest: raise ValueError(f'artifact hash mismatch: {relative}')
    if sha256_file(root/'predictions.jsonl') != manifest['predictions_hash']: raise ValueError('predictions hash mismatch')
    if sha256_file(root/'metrics.json') != manifest['metrics_hash']: raise ValueError('metrics hash mismatch')
    if sha256_file(root/'causal_diagnostics.json') != manifest['causal_diagnostics_hash']: raise ValueError('diagnostics hash mismatch')
    if semantic_run_hash(manifest) != manifest['run_hash']: raise ValueError('run hash mismatch')
    normalization_payload=json.loads((root/'normalization.json').read_text(encoding='utf-8'))
    normalizer=Normalizer.from_dict(normalization_payload)
    if normalizer.semantic_hash != manifest['normalization_hash']: raise ValueError('normalization hash mismatch')
    comparison_path=root/'comparison_lightgbm.json'
    if not comparison_path.exists() or semantic_json_hash(json.loads(comparison_path.read_text(encoding='utf-8'))) != manifest['comparison_hash']:
        raise ValueError('comparison hash mismatch')
    checkpoint=torch.load(root/'model'/'best_model.pt',map_location='cpu',weights_only=False)
    expected_architecture=manifest['architecture']
    if checkpoint.get('architecture', {}) != {key: expected_architecture[key] for key in ('input_channels','hidden_channels','kernel_size','dilations','convs_per_block','dropout','output_channels')}:
        raise ValueError('checkpoint architecture mismatch')
    if checkpoint.get('receptive_field') != expected_architecture['receptive_field'] or checkpoint.get('parameter_count') != expected_architecture['parameter_count']:
        raise ValueError('checkpoint metadata mismatch')
    model=PenTCN(); model.load_state_dict(checkpoint['state_dict']); model.eval(); state_hash=model_state_hash(model)
    if state_hash != manifest['model_state_hash']: raise ValueError('model state hash mismatch')
    rows=_read_predictions(root/'predictions.jsonl'); val=evaluate(model,make_loader(data.by_split['validation'],32),normalizer,.5); test=evaluate(model,make_loader(data.by_split['test'],32),normalizer,.5)
    fresh=val.predictions+test.predictions; saved={r['episode_id']:r for r in rows if r['split'] in ('validation','test')}
    diffs={eid:float(np.max(np.abs(np.asarray(row['probability_vector'])-np.asarray(saved[eid]['probability_vector'])))) for row in fresh for eid in [row['episode_id']]}
    split_max={split:max((diffs[row['episode_id']] for row in fresh if row['split']==split),default=0.) for split in ('validation','test')}
    if max(split_max.values()) > 1e-12: raise ValueError('reload probability mismatch')
    return {'manifest':manifest,'validation_max_abs_diff':split_max['validation'],'test_max_abs_diff':split_max['test'],'per_episode_max_abs_diff':diffs,'status':'PASS'}


def run_baseline(features, output, config, lightgbm_run, overwrite=False):
    root=Path(output)
    if root.exists() and any(root.iterdir()):
        if not overwrite: raise FileExistsError(root)
        shutil.rmtree(root)
    data=load_tcn_dataset(features); run1=execute_training_once(data,root,config); comparison=compare_with_lightgbm(run1,lightgbm_run); write_json(root/'comparison_lightgbm.json',comparison); manifest=_write_manifest(run1,comparison)
    reload=reload_tcn_run(root,features); write_json(root/'audits'/'model_reload.json',reload)
    with tempfile.TemporaryDirectory(prefix='r1-tcn-run2-') as temp:
        run2=execute_training_once(data,temp,config); comparison2=compare_with_lightgbm(run2,lightgbm_run); write_json(Path(temp)/'comparison_lightgbm.json',comparison2); manifest2=_write_manifest(run2,comparison2)
    repro={'run1_prediction_hash':manifest['predictions_hash'],'run2_prediction_hash':manifest2['predictions_hash'],'run1_metrics_hash':manifest['metrics_hash'],'run2_metrics_hash':manifest2['metrics_hash'],'run1_causal_diagnostics_hash':manifest['causal_diagnostics_hash'],'run2_causal_diagnostics_hash':manifest2['causal_diagnostics_hash'],'run1_model_state_hash':manifest['model_state_hash'],'run2_model_state_hash':manifest2['model_state_hash'],'run1_comparison_hash':manifest['comparison_hash'],'run2_comparison_hash':manifest2['comparison_hash'],'run1_run_hash':manifest['run_hash'],'run2_run_hash':manifest2['run_hash'],'run1_model_file_hash':manifest['model_file_hash'],'run2_model_file_hash':manifest2['model_file_hash'],'semantic_reproducibility':all(manifest[k]==manifest2[k] for k in ('predictions_hash','metrics_hash','causal_diagnostics_hash','model_state_hash','comparison_hash','run_hash')),'artifact_byte_reproducibility':manifest['model_file_hash']==manifest2['model_file_hash']}
    repro['status']='PASS' if repro['semantic_reproducibility'] else 'FAIL'; write_json(root/'audits'/'reproducibility.json',repro); return {'manifest':manifest,'comparison':comparison,'reload':reload,'reproducibility':repro}
