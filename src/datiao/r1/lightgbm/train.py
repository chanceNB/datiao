"""Fixed, single-run R1 LightGBM OVR baseline."""
from __future__ import annotations
import argparse, hashlib, json, platform, shutil, tempfile
from pathlib import Path
import numpy as np
import lightgbm as lgb
import sklearn, scipy
from .config import load_config, LightGBMConfig
from .data import load_feature_matrices, input_integrity_audit
from .metrics import calculate_metrics
from .model import fit_predictor, ConstantBinaryPredictor, LightGBMBinaryPredictor
from .manifest import sha256_file, write_json, write_jsonl, run_hash
from ..features.models import FEATURE_ORDER, LABEL_ORDER

def _predictions(data, predictors, threshold):
    rows=[]
    for split in ('validation','test','train'):
        probs=np.column_stack([predictors[label].predict_proba(data.X[split]) for label in LABEL_ORDER])
        preds=(probs>=threshold).astype(int)
        for i,eid in enumerate(data.episode_ids[split]):
            rows.append({'episode_id':eid,'split':split,'probability_vector':probs[i].tolist(),'prediction_vector':preds[i].tolist(),'probabilities':dict(zip(LABEL_ORDER,probs[i].tolist())),'predictions':dict(zip(LABEL_ORDER,preds[i].tolist())),'target':data.Y[split][i].tolist()})
    return sorted(rows,key=lambda x:(x['split'],x['episode_id']))

def _fit_all(data, config, model_dir=None):
    predictors={}; summaries={}; importances={}
    for i,label in enumerate(LABEL_ORDER):
        path=None if model_dir is None else model_dir/(label+'.constant.json' if len(np.unique(data.Y['train'][:,i]))<2 else label+'.txt')
        predictor=fit_predictor(label,data.X['train'],data.Y['train'][:,i],data.X['validation'],data.Y['validation'][:,i],config,path)
        predictors[label]=predictor
        summaries[label]={'label':label,'learner_type':predictor.learner_type,'best_iteration':predictor.best_iteration,'train_positive':int(data.Y['train'][:,i].sum()),'train_negative':int(len(data.Y['train'])-data.Y['train'][:,i].sum()),'validation_positive':int(data.Y['validation'][:,i].sum()),'validation_negative':int(len(data.Y['validation'])-data.Y['validation'][:,i].sum()),'test_positive':int(data.Y['test'][:,i].sum()),'test_negative':int(len(data.Y['test'])-data.Y['test'][:,i].sum()),'feature_order':list(FEATURE_ORDER),'seed':config.seed}
        if predictor.learner_type=='LIGHTGBM':
            importances[label]={'gain':predictor.booster.feature_importance(importance_type='gain').astype(float).tolist(),'split':predictor.booster.feature_importance(importance_type='split').astype(float).tolist(),'feature_order':list(FEATURE_ORDER)}
        else: importances[label]={'status':'NOT_APPLICABLE','feature_order':list(FEATURE_ORDER)}
    return predictors,summaries,importances

def run_baseline(features: str|Path, output: str|Path, config: LightGBMConfig, overwrite=False):
    root=Path(output)
    if root.exists() and any(root.iterdir()):
        if not overwrite: raise FileExistsError(f'output exists: {root}')
        shutil.rmtree(root)
    root.mkdir(parents=True,exist_ok=True); data=load_feature_matrices(features); audit=input_integrity_audit(data)
    if audit['status']!='PASS': raise ValueError(audit)
    write_json(root/'audits'/'input_integrity.json',audit)
    model_dir=root/'models'; model_dir.mkdir(parents=True)
    predictors,summaries,importances=_fit_all(data,config,model_dir)
    rows=_predictions(data,predictors,config.threshold); write_jsonl(root/'predictions.jsonl',rows)
    metrics={split:calculate_metrics(data.Y[split],np.asarray([[r['probabilities'][label] for label in LABEL_ORDER] for r in rows if r['split']==split]),config.threshold) for split in ('validation','test')}
    write_json(root/'metrics.json',metrics); write_json(root/'label_summary.json',summaries); write_json(root/'feature_importance.json',importances)
    config_payload=config.model_dump(mode='json'); write_json(root/'config_snapshot.json',config_payload)
    # Reload every persisted learner and require probability identity.
    reloaded={}
    for label,predictor in predictors.items():
        summary=summaries[label]; path=model_dir/(label+'.constant.json' if summary['learner_type']!='LIGHTGBM' else label+'.txt')
        reloaded[label]=ConstantBinaryPredictor.load(path) if summary['learner_type']!='LIGHTGBM' else LightGBMBinaryPredictor.load(path,label,summary['best_iteration'])
    original=np.column_stack([predictors[label].predict_proba(data.X['test']) for label in LABEL_ORDER]); loaded=np.column_stack([reloaded[label].predict_proba(data.X['test']) for label in LABEL_ORDER]); max_diff=float(np.max(np.abs(original-loaded)))
    write_json(root/'audits'/'reproducibility.json',{'same_config':True,'same_prediction_hash':True,'same_metrics_hash':True,'per_label_probability_max_abs_diff':max_diff,'status':'PASS' if max_diff<=1e-12 else 'FAIL'})
    if max_diff>1e-12: raise ValueError('reloaded model probabilities differ')
    model_hashes={p.name:sha256_file(p) for p in model_dir.iterdir()}; metrics_hash=sha256_file(root/'metrics.json'); predictions_hash=sha256_file(root/'predictions.jsonl')
    manifest={'run_version':'1.0.0','baseline_id':config.baseline_id,'baseline_version':config.baseline_version,'source_feature_dataset_id':data.dataset['manifest'].feature_dataset_id,'source_feature_dataset_version':data.dataset['manifest'].feature_dataset_version,'source_feature_manifest_hash':data.dataset['manifest'].manifest_hash,'source_dataset_manifest_hash':data.dataset['manifest'].source_dataset_manifest_hash,'source_split_manifest_hash':data.dataset['manifest'].source_split_manifest_hash,'feature_schema_version':data.dataset['manifest'].feature_schema_version,'target_spec_version':data.dataset['manifest'].target_spec_version,'feature_builder_version':data.dataset['manifest'].feature_builder_version,'truth_alignment_version':data.dataset['manifest'].truth_alignment_version,'feature_order':list(FEATURE_ORDER),'label_order':list(LABEL_ORDER),'train_count':len(data.X['train']),'validation_count':len(data.X['validation']),'test_count':len(data.X['test']),'seed':config.seed,'threshold':config.threshold,'config_hash':f"sha256:{hashlib.sha256(json.dumps(config_payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()}",'label_learners':summaries,'model_file_hashes':model_hashes,'predictions_hash':predictions_hash,'metrics_hash':metrics_hash}
    manifest['run_hash']=run_hash(manifest); write_json(root/'run_manifest.json',manifest)
    env={'python':platform.python_version(),'lightgbm':lgb.__version__,'numpy':np.__version__,'scikit_learn':sklearn.__version__,'scipy':scipy.__version__,'platform':platform.platform()}; write_json(root/'environment.json',env)
    # A second in-memory fit checks deterministic probabilities without touching test during fit or tuning.
    second,_,_=_fit_all(data,config,None); second_rows=_predictions(data,second,config.threshold); second_hash=sha256_file(root/'predictions.jsonl') if second_rows==rows else 'DIFF'
    repro= json.loads((root/'audits'/'reproducibility.json').read_text()); repro.update({'same_prediction_hash':second_hash!='DIFF','same_metrics_hash':second_hash!='DIFF'}); write_json(root/'audits'/'reproducibility.json',repro)
    if second_hash=='DIFF': raise ValueError('second-run predictions differ')
    return manifest

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--features',required=True); ap.add_argument('--output',required=True); ap.add_argument('--config',required=True); ap.add_argument('--overwrite',action='store_true'); args=ap.parse_args(argv)
    result=run_baseline(args.features,args.output,load_config(args.config),args.overwrite); print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
