"""Frozen SequenceSample adapter; IDs and metadata never become input channels."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..features.io import reload_feature_dataset
from ..features.models import FEATURE_ORDER, LABEL_ORDER
from .manifest import semantic_json_hash

FEATURE_MANIFEST = 'sha256:40b0326baf0f71206da1fd14b05c8ab5bdf1473c708735c57c019ea820cf5848'
SPLITS = ('train', 'validation', 'test')
CAUSAL_LABELS = ('WRITING','QUESTION_VISIT','RETURN','REVISION_CANDIDATE','PAGE_CHANGE','UNKNOWN')


@dataclass
class TCNDataset:
    manifest: object
    sequences: tuple
    episodes: tuple
    by_split: dict
    integrity: dict
    source_path: str


def load_tcn_dataset(features: str | Path) -> TCNDataset:
    raw = reload_feature_dataset(features)
    manifest = raw['manifest']
    if manifest.manifest_hash != FEATURE_MANIFEST:
        raise ValueError('frozen Feature Manifest mismatch')
    sequences, episodes = raw['sequences'], raw['episodes']
    by_split = {split: tuple(s for s in sequences if s.split == split) for split in SPLITS}
    counts = {split: {'sequences':len(by_split[split]), 'episodes':sum(len(s.values) for s in by_split[split])} for split in SPLITS}
    expected = {'train':{'sequences':252,'episodes':490}, 'validation':{'sequences':54,'episodes':105}, 'test':{'sequences':54,'episodes':105}}
    if counts != expected or len(sequences)!=360 or len(episodes)!=700:
        raise ValueError('frozen counts mismatch')
    episode_map = {e.episode_id: e for e in episodes}
    rows = {r.episode_id: r for r in raw['feature_rows']}
    owned = []
    participant_splits = {}
    for sequence in sequences:
        participant_splits.setdefault(sequence.participant_id, set()).add(sequence.split)
        for i, eid in enumerate(sequence.episode_ids):
            owned.append(eid)
            ep = episode_map.get(eid)
            if ep is None or ep.split != sequence.split or ep.participant_id != sequence.participant_id:
                raise ValueError('episode split/participant inheritance mismatch')
            if tuple(sequence.targets[i]) != rows[eid].target or any(x not in (0,1) for x in sequence.targets[i]):
                raise ValueError('sequence target mismatch')
            for j, feature in enumerate(FEATURE_ORDER):
                value = rows[eid].features[feature]
                if sequence.feature_mask[i][j] != int(value is not None) or sequence.values[i][j] != (0. if value is None else value):
                    raise ValueError('sequence values/mask mismatch')
    if len(set(owned)) != len(owned) or set(owned) != set(episode_map):
        raise ValueError('episode ownership must be exactly once')
    if any(len(split_set)!=1 for split_set in participant_splits.values()):
        raise ValueError('participant split overlap')
    integrity = {'status':'PASS','source_feature_manifest_hash':FEATURE_MANIFEST,'sequence_count':len(sequences), 'real_timestep_count':len(owned),'split_counts':counts,'feature_dimension':12,'label_dimension':8,'feature_masks_binary':True,'targets_binary':True,'episode_ownership_unique':True,'split_inheritance':True,'no_lightgbm_input':True,'no_metadata_channel':True,'input_fields':['values','feature_mask']}
    return TCNDataset(manifest, sequences, episodes, by_split, integrity, str(features))


@dataclass(frozen=True)
class Normalizer:
    mean: tuple
    std: tuple
    observed_count: tuple
    zero_variance_features: tuple
    fit_split: str = 'train'

    def to_dict(self):
        return {'feature_order':list(FEATURE_ORDER),'mean':list(self.mean),'std':list(self.std),'observed_count':list(self.observed_count),'zero_variance_features':list(self.zero_variance_features),'fit_split':self.fit_split}

    @property
    def semantic_hash(self):
        return semantic_json_hash(self.to_dict())

    @classmethod
    def from_dict(cls, payload):
        if payload['feature_order'] != list(FEATURE_ORDER) or payload['fit_split'] != 'train':
            raise ValueError('normalization lineage mismatch')
        if any(len(payload[k]) != 12 for k in ('mean','std','observed_count')) or not np.isfinite(payload['mean']).all() or not np.isfinite(payload['std']).all() or np.any(np.asarray(payload['std'])<=0):
            raise ValueError('invalid normalization statistics')
        return cls(tuple(payload['mean']),tuple(payload['std']),tuple(payload['observed_count']),tuple(payload['zero_variance_features']))

    def transform(self, values, feature_mask):
        means = torch.as_tensor(self.mean, dtype=values.dtype, device=values.device)
        scales = torch.as_tensor(self.std, dtype=values.dtype, device=values.device)
        return torch.where(feature_mask.bool(), (values-means)/scales, torch.zeros_like(values))


def fit_normalizer(train_sequences) -> Normalizer:
    train = [s for s in train_sequences if s.split == 'train']
    if not train:
        raise ValueError('Train sequences required')
    values = np.concatenate([s.values for s in train]).astype(np.float64)
    masks = np.concatenate([s.feature_mask for s in train]).astype(bool)
    mean, scale, counts, zero = [], [], [], []
    for j, feature in enumerate(FEATURE_ORDER):
        observed = values[masks[:,j],j]
        if not np.isfinite(observed).all():
            raise ValueError('non-finite Train observation')
        count = len(observed)
        std = float(observed.std()) if count else 0.
        mean.append(float(observed.mean()) if count else 0.)
        if std <= 1e-12:
            zero.append(feature)
        scale.append(1. if std <= 1e-12 else std)
        counts.append(count)
    return Normalizer(tuple(mean),tuple(scale),tuple(counts),tuple(zero))


def collate_sequences(batch):
    max_t = max(len(s.values) for s in batch)
    values = torch.zeros(len(batch),max_t,12)
    masks = torch.zeros_like(values)
    targets = torch.zeros(len(batch),max_t,8)
    padding = torch.zeros(len(batch),max_t)
    for i, sample in enumerate(batch):
        t = len(sample.values)
        values[i,:t] = torch.tensor(sample.values)
        masks[i,:t] = torch.tensor(sample.feature_mask)
        targets[i,:t] = torch.tensor(sample.targets)
        padding[i,:t] = 1
    return {'values':values,'feature_mask':masks,'targets':targets,'padding_mask':padding,'episode_ids':[s.episode_ids for s in batch],'sequence_ids':[s.sequence_id for s in batch],'splits':[s.split for s in batch]}


def prepare_model_input(values, feature_mask, normalizer):
    return torch.cat((normalizer.transform(values,feature_mask),feature_mask),dim=-1)


def make_loader(sequences, batch_size, *, shuffle=False, seed=20260929):
    return DataLoader(list(sequences),batch_size=batch_size,shuffle=shuffle,num_workers=0,collate_fn=collate_sequences,generator=torch.Generator().manual_seed(seed))


def build_sequence_length_audit(sequences, receptive_field):
    def summarize(selected):
        lengths = np.asarray([len(s.episode_ids) for s in selected],dtype=int)
        n = len(lengths)
        count_rf = int(np.sum(lengths>=receptive_field))
        return {'sequence_count':n,'real_timestep_count':int(lengths.sum()),'min':int(lengths.min()) if n else None,'mean':float(lengths.mean()) if n else None,'median':float(np.median(lengths)) if n else None,'p90':float(np.percentile(lengths,90)) if n else None,'max':int(lengths.max()) if n else None,'count_T_eq_1':int(np.sum(lengths==1)),'ratio_T_eq_1':float(np.mean(lengths==1)) if n else 0.,'count_T_ge_2':int(np.sum(lengths>=2)),'ratio_T_ge_2':float(np.mean(lengths>=2)) if n else 0.,'count_T_ge_3':int(np.sum(lengths>=3)),'ratio_T_ge_3':float(np.mean(lengths>=3)) if n else 0.,'count_T_ge_4':int(np.sum(lengths>=4)),'ratio_T_ge_4':float(np.mean(lengths>=4)) if n else 0.,'sequence_count_length_ge_RF':count_rf,'ratio_length_ge_RF':count_rf/n if n else 0.,'effective_temporal_context_warning':'SHORT_SEQUENCE_LIMITED_CONTEXT' if n and count_rf/n < .5 else None}
    return {'status':'PASS','receptive_field':receptive_field,'warning_policy':'fraction reaching RF < 0.5; descriptive only, never tuning','overall':summarize(sequences),**{split:summarize([s for s in sequences if s.split==split]) for split in SPLITS}}


def target_causality_audit():
    rules = {
        'WRITING':('Current action pen geometry/contact','current action point-ref episode',False,False,False),
        'QUESTION_VISIT':('Current question mapping and visited/active state','current action point-ref episode',True,False,False),
        'QUESTION_LEAVE':('Next question/page or unavailable mapping establishes leave','previous action point-ref episode',True,True,False),
        'RETURN':('Current mapping to previously visited question','current action point-ref episode',True,False,False),
        'REVISION_CANDIDATE':('Scenario-plan revision action; detector uses prior ink, pause/revisit and overlap','current revision action point-ref episode',True,False,False),
        'PAGE_CHANGE':('Current page differs from previous page','current action point-ref episode',True,False,False),
        'PROCESS_END':('Explicit end signal or external session-end timeout reference','final action point-ref episode',False,False,True),
        'UNKNOWN':('Scenario expected unknown mapping; current mapping/quality observation','current action point-ref episode',False,False,False),
    }
    labels = []
    for label in LABEL_ORDER:
        rule, anchor, past, future, external = rules[label]
        category = 'RETROSPECTIVE_TRANSITION' if future else 'EXTERNAL_SIGNAL_DEPENDENT' if external else 'CAUSAL_OBSERVABLE'
        note = 'Retrospectively attributed; not observable at anchored timestep.' if future else 'External signal absent from values/mask.' if external else 'Current/historical observables; compressed features may not fully represent the rule.'
        if label=='WRITING':
            note += ' All-positive; AUC/AP undefined; no evidence of discriminative learning.'
        labels.append({'label':label,'truth_generation_rule':rule,'truth_anchor_episode':anchor,'information_required':rule,'current_timestep_sufficient':not future and not external,'past_context_required':past,'future_timestep_required':future,'external_signal_required':external,'causality_class':category,'notes':note})
    return {'labels':labels,'causal_observable_label_names':list(CAUSAL_LABELS),'retrospective_labels':['QUESTION_LEAVE'],'external_signal_labels':['PROCESS_END'],'status':'PASS_WITH_NON_CAUSAL_LABELS'}
