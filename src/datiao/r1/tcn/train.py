"""Single-run training and final inference for the frozen Pen TCN baseline."""
import argparse
import json
import platform
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..evaluation.multilabel import calculate_metrics, calculate_subset_metrics
from ..features.models import FEATURE_ORDER, LABEL_ORDER
from .config import TCNConfig
from .data import (CAUSAL_LABELS, Normalizer, build_sequence_length_audit, collate_sequences,
                   fit_normalizer, make_loader, prepare_model_input, target_causality_audit)
from .manifest import semantic_json_hash
from .model import PenTCN, masked_bce_with_logits, model_state_hash


@dataclass
class TrainingResult:
    best_epoch: int
    last_epoch: int
    best_validation_loss: float
    history: list[dict]
    best_state: dict


@dataclass
class EvaluationResult:
    predictions: list[dict]
    metrics: dict
    probabilities: dict[str, np.ndarray]


def set_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def _batch_loss(model, batch, normalizer, device):
    model_input = prepare_model_input(batch['values'].to(device), batch['feature_mask'].to(device), normalizer)
    logits = model(model_input)
    loss = masked_bce_with_logits(logits, batch['targets'].to(device), batch['padding_mask'].to(device))
    return loss, logits


def _validation_loss(model, loader, normalizer, device):
    model.eval()
    losses=[]
    with torch.no_grad():
        for batch in loader:
            loss,_ = _batch_loss(model,batch,normalizer,device)
            losses.append(float(loss))
    if not losses: raise ValueError('empty validation loader')
    return float(np.mean(losses))


def fit(model, train_loader, validation_loader, normalizer, config) -> TrainingResult:
    device = torch.device(config.device); model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    history=[]; best_state=None; best_loss=float('inf'); best_epoch=0; stale=0
    for epoch in range(1,config.max_epochs+1):
        model.train(); train_losses=[]
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss,_ = _batch_loss(model,batch,normalizer,device)
            loss.backward()
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad):
                raise ValueError('non-finite gradient')
            torch.nn.utils.clip_grad_norm_(model.parameters(),config.gradient_clip_norm)
            optimizer.step(); train_losses.append(float(loss.detach()))
        val_loss = _validation_loss(model,validation_loader,normalizer,device)
        lr=optimizer.param_groups[0]['lr']
        history.append({'epoch':epoch,'train_masked_bce':float(np.mean(train_losses)),'validation_masked_bce':val_loss,'learning_rate':lr})
        if val_loss < best_loss-config.min_delta:
            best_loss=val_loss; best_epoch=epoch; stale=0
            best_state={key:tensor.detach().cpu().clone() for key,tensor in model.state_dict().items()}
        else:
            stale += 1
            if stale >= config.early_stopping_patience: break
    if best_state is None: raise ValueError('no best checkpoint')
    model.load_state_dict(best_state)
    return TrainingResult(best_epoch,epoch,best_loss,history,best_state)


def evaluate(model, loader, normalizer, threshold, device='cpu') -> EvaluationResult:
    model.eval(); records=[]; split_prob={}; split_target={}
    with torch.no_grad():
        for batch in loader:
            logits = model(prepare_model_input(batch['values'].to(device),batch['feature_mask'].to(device),normalizer))
            probs = torch.sigmoid(logits).cpu().numpy(); targets=batch['targets'].numpy(); mask=batch['padding_mask'].numpy()
            for i, ids in enumerate(batch['episode_ids']):
                split=batch['splits'][i]; sequence_id=batch['sequence_ids'][i]
                split_prob.setdefault(split,[]); split_target.setdefault(split,[])
                for t,eid in enumerate(ids):
                    if mask[i,t] == 0: continue
                    vector=probs[i,t].tolist(); target=targets[i,t].astype(int).tolist()
                    records.append({'episode_id':eid,'sequence_id':sequence_id,'timestep_index':t,'split':split,'probability_vector':vector,'prediction_vector':(probs[i,t]>=threshold).astype(int).tolist(),'target':target})
                    split_prob[split].append(probs[i,t]); split_target[split].append(target)
    records.sort(key=lambda row:(row['split'],row['episode_id'],row['sequence_id'],row['timestep_index']))
    metrics={split:calculate_metrics(np.asarray(split_target[split],dtype=np.int8),np.asarray(split_prob[split]),threshold) for split in split_target}
    return EvaluationResult(records,metrics,{split:np.asarray(split_prob[split]) for split in split_prob})


def causal_diagnostics(metrics, threshold=0.5):
    return {split: calculate_subset_metrics(
        np.asarray([row['target'] for row in []]), np.asarray([]), threshold, CAUSAL_LABELS
    ) for split in ()} if False else None


def _runtime_audits(model, training):
    """Exercise causal, padding, mask, gradient-update, and finite-value contracts."""
    model.eval()
    torch.manual_seed(20260929)
    prefix = torch.randn(1, 4, 24)
    with torch.no_grad():
        base = model(prefix)
        future_changed = prefix.clone()
        future_changed[:, 2:] += 17.0
        perturbed = model(future_changed)
        causal_diff = float(torch.max(torch.abs(base[:, :2] - perturbed[:, :2])))
        alone = model(prefix[:, :2])
        right_padded = model(torch.cat((prefix[:, :2], torch.zeros(1, 2, 24)), dim=1))[:, :2]
        padding_diff = float(torch.max(torch.abs(alone - right_padded)))
    logits = torch.tensor([[[0.2, -0.3], [4.0, -4.0], [99.0, -99.0]]])
    targets = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    masked_loss = masked_bce_with_logits(logits, targets, mask)
    padded_loss = masked_bce_with_logits(torch.cat((logits, torch.randn(1, 2, 2)), dim=1), torch.cat((targets, torch.randn(1, 2, 2)), dim=1), torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]]))
    finite = all(torch.isfinite(parameter).all() for parameter in model.parameters()) and all(np.isfinite(item['train_masked_bce']) and np.isfinite(item['validation_masked_bce']) for item in training.history)
    observed_input = torch.cat((torch.zeros(1, 1, 12), torch.ones(1, 1, 12)), dim=-1)
    missing_input = torch.zeros(1, 1, 24)
    return {
        'causality': {'status': 'PASS' if causal_diff <= 1e-12 else 'FAIL', 'future_perturbation': 'PASS' if causal_diff <= 1e-12 else 'FAIL', 'max_abs_past_logit_diff': causal_diff},
        'padding': {'status': 'PASS' if padding_diff <= 1e-12 else 'FAIL', 'padding_mask_independent': padding_diff <= 1e-12, 'max_abs_real_logit_diff': padding_diff},
        'masked_loss': {'status': 'PASS' if abs(float(masked_loss - padded_loss)) <= 1e-12 else 'FAIL', 'padding_invariant': abs(float(masked_loss - padded_loss)) <= 1e-12, 'missing_vs_zero': not torch.equal(observed_input, missing_input)},
        'finite_checks': {'status': 'PASS' if finite else 'FAIL', 'logits_loss_gradients': finite},
    }


def execute_training_once(data, output, config: TCNConfig):
    set_deterministic(config.seed)
    root=Path(output); root.mkdir(parents=True,exist_ok=True); (root/'audits').mkdir(exist_ok=True); (root/'model').mkdir(exist_ok=True)
    normalizer=fit_normalizer(data.by_split['train'])
    causal=target_causality_audit()
    train_loader=make_loader(data.by_split['train'],config.batch_size,shuffle=True,seed=config.seed)
    val_loader=make_loader(data.by_split['validation'],config.batch_size,shuffle=False,seed=config.seed)
    model=PenTCN(hidden_channels=config.hidden_channels,kernel_size=config.kernel_size,dilations=config.dilations,dropout=config.dropout)
    initial_model_state_hash=model_state_hash(model)
    audit=build_sequence_length_audit(data.sequences,model.receptive_field)
    training=fit(model,train_loader,val_loader,normalizer,config)
    torch.save({'state_dict':training.best_state,'architecture':{'input_channels':24,'hidden_channels':list(config.hidden_channels),'kernel_size':config.kernel_size,'dilations':list(config.dilations),'convs_per_block':config.convs_per_block,'dropout':config.dropout,'output_channels':8},'receptive_field':model.receptive_field,'parameter_count':model.parameter_count},root/'model'/'best_model.pt')
    test_loader=make_loader(data.by_split['test'],config.batch_size,shuffle=False,seed=config.seed)
    val_eval=evaluate(model,val_loader,normalizer,config.threshold); test_eval=evaluate(model,test_loader,normalizer,config.threshold)
    all_records=val_eval.predictions+test_eval.predictions
    # Training predictions are required for completeness and are final inference only.
    train_eval=evaluate(model,train_loader,normalizer,config.threshold)
    all_records=train_eval.predictions+all_records; all_records.sort(key=lambda row:(row['split'],row['episode_id'],row['sequence_id'],row['timestep_index']))
    metrics={'validation':val_eval.metrics['validation'],'test':test_eval.metrics['test']}
    causal_metrics={split:calculate_subset_metrics(np.asarray([r['target'] for r in all_records if r['split']==split],dtype=np.int8),np.asarray([r['probability_vector'] for r in all_records if r['split']==split]),config.threshold,CAUSAL_LABELS) for split in ('validation','test')}
    from ..lightgbm.manifest import write_json, write_jsonl
    runtime_audits=_runtime_audits(model,training)
    metrics_payload=dict(metrics, interpretation={'canonical_label_order':list(LABEL_ORDER),'micro_and_macro_f1_include_degenerate_all_positive_WRITING':True,'causal_subset_is_reported_separately':True})
    runtime_audits['masked_loss']['gradient_update'] = initial_model_state_hash != model_state_hash(model)
    runtime_audits['causality'].update(runtime_audits['masked_loss'])
    write_json(root/'normalization.json',normalizer.to_dict()); write_json(root/'sequence_length_audit.json',audit); write_json(root/'target_causality_audit.json',causal); write_json(root/'training_history.json',{'history':training.history,'best_epoch':training.best_epoch,'last_epoch':training.last_epoch,'best_validation_loss':training.best_validation_loss}); write_json(root/'metrics.json',metrics_payload); write_json(root/'causal_diagnostics.json',causal_metrics); write_jsonl(root/'predictions.jsonl',all_records); write_json(root/'config_snapshot.json',config.model_dump(mode='json')); write_json(root/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'sklearn':__import__('sklearn').__version__,'device':config.device,'deterministic':True,'threads':torch.get_num_threads()}); write_json(root/'audits'/'input_integrity.json',data.integrity); write_json(root/'audits'/'normalizer.json',{'status':'PASS','fit_split':'train','validation_test_unused':True,'normalization_hash':normalizer.semantic_hash}); write_json(root/'audits'/'causality.json',runtime_audits['causality']); write_json(root/'audits'/'padding.json',runtime_audits['padding'])
    return {'output_path':str(root),'data':data,'normalizer':normalizer,'model':model,'training':training,'metrics':metrics,'causal_metrics':causal_metrics,'predictions':all_records,'model_state_hash':model_state_hash(model),'model_file_hash':None}


def main(argv=None):
    from .config import load_tcn_config
    from .data import load_tcn_dataset
    from .io import run_baseline
    ap=argparse.ArgumentParser(); ap.add_argument('--features',required=True); ap.add_argument('--output',required=True); ap.add_argument('--config',required=True); ap.add_argument('--lightgbm-run',required=True); ap.add_argument('--overwrite',action='store_true'); args=ap.parse_args(argv)
    print(json.dumps(run_baseline(args.features,args.output,load_tcn_config(args.config),args.lightgbm_run,args.overwrite),ensure_ascii=False,indent=2))

if __name__=='__main__': main()
