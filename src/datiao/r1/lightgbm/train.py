"""Fixed, reproducible R1 LightGBM OVR baseline."""
from __future__ import annotations
import argparse, hashlib, json, platform, shutil, tempfile
from pathlib import Path
import numpy as np
import lightgbm as lgb
import sklearn, scipy
from .config import load_config, LightGBMConfig
from .data import FeatureMatrices, load_feature_matrices, input_integrity_audit
from .metrics import calculate_metrics
from .model import fit_predictor, ConstantBinaryPredictor, LightGBMBinaryPredictor
from .manifest import sha256_file, write_json, write_jsonl, run_hash
from ..features.models import FEATURE_ORDER, LABEL_ORDER

def _probabilities(data, predictors):
    return {split: np.column_stack([predictors[label].predict_proba(data.X[split]) for label in LABEL_ORDER]) for split in ('train','validation','test')}

def _predictions(data, probabilities, threshold):
    rows=[]
    for split in ('validation','test','train'):
        probs=probabilities[split]; preds=(probs>=threshold).astype(int)
        for i,eid in enumerate(data.episode_ids[split]):
            rows.append({'episode_id':eid,'split':split,'probability_vector':probs[i].tolist(),'prediction_vector':preds[i].tolist(),'probabilities':dict(zip(LABEL_ORDER,probs[i].tolist())),'predictions':dict(zip(LABEL_ORDER,preds[i].tolist())),'target':data.Y[split][i].tolist()})
    return sorted(rows,key=lambda x:(x['split'],x['episode_id']))

def _fit_all(data, config, model_dir=None):
    predictors={}; summaries={}; importances={}
    for i,label in enumerate(LABEL_ORDER):
        is_constant=len(np.unique(data.Y['train'][:,i]))<2
        path=None if model_dir is None else model_dir/(label+('.constant.json' if is_constant else '.txt'))
        predictor=fit_predictor(label,data.X['train'],data.Y['train'][:,i],data.X['validation'],data.Y['validation'][:,i],config,path)
        predictors[label]=predictor
        summaries[label]={'label':label,'learner_type':predictor.learner_type,'best_iteration':predictor.best_iteration,'train_positive':int(data.Y['train'][:,i].sum()),'train_negative':int(len(data.Y['train'])-data.Y['train'][:,i].sum()),'validation_positive':int(data.Y['validation'][:,i].sum()),'validation_negative':int(len(data.Y['validation'])-data.Y['validation'][:,i].sum()),'test_positive':int(data.Y['test'][:,i].sum()),'test_negative':int(len(data.Y['test'])-data.Y['test'][:,i].sum()),'feature_order':list(FEATURE_ORDER),'seed':config.seed}
        importances[label]=({'gain':predictor.booster.feature_importance(importance_type='gain').astype(float).tolist(),'split':predictor.booster.feature_importance(importance_type='split').astype(float).tolist(),'feature_order':list(FEATURE_ORDER)} if predictor.learner_type=='LIGHTGBM' else {'status':'NOT_APPLICABLE','feature_order':list(FEATURE_ORDER)})
    return predictors,summaries,importances

def _reload_predictors(model_dir,summaries):
    loaded={}
    for label,summary in summaries.items():
        path=model_dir/(label+('.constant.json' if summary['learner_type']!='LIGHTGBM' else '.txt'))
        loaded[label]=ConstantBinaryPredictor.load(path) if summary['learner_type']!='LIGHTGBM' else LightGBMBinaryPredictor.load(path,label,summary['best_iteration'])
    return loaded

def _manifest_payload(data,config,summaries,model_hashes,predictions_hash,metrics_hash):
    config_payload=config.model_dump(mode='json')
    return {'run_version':'1.0.0','baseline_id':config.baseline_id,'baseline_version':config.baseline_version,'source_feature_dataset_id':data.dataset['manifest'].feature_dataset_id,'source_feature_dataset_version':data.dataset['manifest'].feature_dataset_version,'source_feature_manifest_hash':data.dataset['manifest'].manifest_hash,'source_dataset_manifest_hash':data.dataset['manifest'].source_dataset_manifest_hash,'source_split_manifest_hash':data.dataset['manifest'].source_split_manifest_hash,'feature_schema_version':data.dataset['manifest'].feature_schema_version,'target_spec_version':data.dataset['manifest'].target_spec_version,'feature_builder_version':data.dataset['manifest'].feature_builder_version,'truth_alignment_version':data.dataset['manifest'].truth_alignment_version,'feature_order':list(FEATURE_ORDER),'label_order':list(LABEL_ORDER),'train_count':len(data.X['train']),'validation_count':len(data.X['validation']),'test_count':len(data.X['test']),'seed':config.seed,'threshold':config.threshold,'config_hash':f"sha256:{hashlib.sha256(json.dumps(config_payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()}",'label_learners':summaries,'model_file_hashes':model_hashes,'predictions_hash':predictions_hash,'metrics_hash':metrics_hash}

def execute_training_once(data: FeatureMatrices, config: LightGBMConfig, output: str|Path):
    root=Path(output); root.mkdir(parents=True,exist_ok=True); model_dir=root/'models'; model_dir.mkdir(parents=True,exist_ok=True)
    predictors,summaries,importances=_fit_all(data,config,model_dir); probabilities=_probabilities(data,predictors)
    metrics={split:calculate_metrics(data.Y[split],probabilities[split],config.threshold) for split in ('validation','test')}; rows=_predictions(data,probabilities,config.threshold)
    write_json(root/'metrics.json',metrics); write_jsonl(root/'predictions.jsonl',rows); write_json(root/'label_summary.json',summaries); write_json(root/'feature_importance.json',importances); write_json(root/'config_snapshot.json',config.model_dump(mode='json'))
    reloaded=_reload_predictors(model_dir,summaries); reload_probabilities=_probabilities(data,reloaded)
    vd={label:float(np.max(np.abs(probabilities['validation'][:,i]-reload_probabilities['validation'][:,i]))) for i,label in enumerate(LABEL_ORDER)}; td={label:float(np.max(np.abs(probabilities['test'][:,i]-reload_probabilities['test'][:,i]))) for i,label in enumerate(LABEL_ORDER)}
    reload_audit={'validation_max_abs_diff':max(vd.values()),'test_max_abs_diff':max(td.values()),'per_label_validation_max_abs_diff':vd,'per_label_test_max_abs_diff':td,'status':'PASS' if max(vd.values())<=1e-12 and max(td.values())<=1e-12 else 'FAIL'}
    write_json(root/'audits'/'model_reload.json',reload_audit)
    if reload_audit['status']!='PASS': raise ValueError('reloaded model probabilities differ')
    model_hashes={path.name:sha256_file(path) for path in sorted(model_dir.iterdir())}; payload=_manifest_payload(data,config,summaries,model_hashes,sha256_file(root/'predictions.jsonl'),sha256_file(root/'metrics.json')); payload['run_hash']=run_hash(payload); write_json(root/'run_manifest.json',payload)
    write_json(root/'environment.json',{'python':platform.python_version(),'lightgbm':lgb.__version__,'numpy':np.__version__,'scikit_learn':sklearn.__version__,'scipy':scipy.__version__,'platform':platform.platform()})
    return {'manifest':payload,'predictions_hash':payload['predictions_hash'],'metrics_hash':payload['metrics_hash'],'run_hash':payload['run_hash'],'reload_audit':reload_audit}

def run_baseline(features: str|Path, output: str|Path, config: LightGBMConfig, overwrite=False):
    root=Path(output)
    if root.exists() and any(root.iterdir()):
        if not overwrite: raise FileExistsError(f'output exists: {root}')
        shutil.rmtree(root)
    root.mkdir(parents=True,exist_ok=True); data=load_feature_matrices(features); audit=input_integrity_audit(data)
    if audit['status']!='PASS': raise ValueError(audit)
    write_json(root/'audits'/'input_integrity.json',audit); run1=execute_training_once(data,config,root)
    with tempfile.TemporaryDirectory(prefix='r1-lightgbm-repro-') as temporary:
        run2=execute_training_once(data,config,Path(temporary)); same_prediction=run1['predictions_hash']==run2['predictions_hash']; same_metrics=run1['metrics_hash']==run2['metrics_hash']; same_run=run1['run_hash']==run2['run_hash']
        repro={'status':'PASS' if same_prediction and same_metrics and same_run else 'FAIL','same_prediction_hash':same_prediction,'same_metrics_hash':same_metrics,'same_run_hash':same_run,'run1_prediction_hash':run1['predictions_hash'],'run2_prediction_hash':run2['predictions_hash'],'run1_metrics_hash':run1['metrics_hash'],'run2_metrics_hash':run2['metrics_hash'],'run1_run_hash':run1['run_hash'],'run2_run_hash':run2['run_hash'],'validation_max_abs_diff':run1['reload_audit']['validation_max_abs_diff'],'test_max_abs_diff':run1['reload_audit']['test_max_abs_diff'],'per_label_validation_max_abs_diff':run1['reload_audit']['per_label_validation_max_abs_diff'],'per_label_test_max_abs_diff':run1['reload_audit']['per_label_test_max_abs_diff']}
    write_json(root/'audits'/'reproducibility.json',repro)
    if repro['status']!='PASS': raise ValueError('reproducibility audit failed')
    return run1['manifest']

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--features',required=True); ap.add_argument('--output',required=True); ap.add_argument('--config',required=True); ap.add_argument('--overwrite',action='store_true'); args=ap.parse_args(argv); print(json.dumps(run_baseline(args.features,args.output,load_config(args.config),args.overwrite),ensure_ascii=False,indent=2))
if __name__=='__main__': main()
