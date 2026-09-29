import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from datiao.r1.dataset import load_dataset_config, materialize_dataset
from datiao.r1.features.builder import (_canonical_contracts, _duration_s, _iou, _norm_path, _pause_before_s, _pause_inside_s,
                                        _question_history_value, materialize_features)
from datiao.r1.features.io import reload_feature_dataset
from datiao.r1.features.alignment import align_truth, target_vector
from datiao.r1.features.models import FEATURE_ORDER, LABEL_ORDER, FeatureRow, ObservationEpisode, SequenceSample
from datiao.r1.synthetic.models import SyntheticTruthEvent


def test_feature_order_target_and_mask_contract():
    row=FeatureRow(episode_id='e', split='train', features={k:(0.0 if k in {'duration_s','quality_valid'} else None) for k in FEATURE_ORDER}, target=(1,0,0,0,0,0,0,0))
    sample=SequenceSample(sequence_id='s',case_id='c',participant_id='p',task_segment_id='t',split='train',episode_ids=('e',),values=((0.0,)*12,),feature_mask=((1,0,0,0,0,0,0,0,0,0,0,1),),targets=(row.target,))
    assert sample.values[0][0] == 0.0 and sample.feature_mask[0][0] == 1
    assert sample.values[0][1] == 0.0 and sample.feature_mask[0][1] == 0


def test_feature_row_rejects_extra_and_missing_keys():
    valid={k:0.0 for k in FEATURE_ORDER}
    with pytest.raises(ValueError): FeatureRow(episode_id='e',split='train',features={**valid,'scenario_type':1.0},target=(0,)*8)
    with pytest.raises(ValueError): FeatureRow(episode_id='e',split='train',features={k:v for k,v in valid.items() if k!='duration_s'},target=(0,)*8)


def test_observable_return_count_is_cumulative_and_unknown_is_neutral():
    history={}
    assert _question_history_value(history,'Q1','MAPPED') == (1,0)
    assert _question_history_value(history,'Q2','MAPPED') == (1,0)
    assert _question_history_value(history,'Q1','MAPPED') == (2,1)
    assert _question_history_value(history,'Q2','MAPPED') == (2,1)
    assert _question_history_value(history,'Q1','MAPPED') == (3,2)
    assert _question_history_value(history,None,'UNKNOWN') == (None,None)


def test_truth_alignment_builds_multilabel_target_without_prediction():
    episode=ObservationEpisode(episode_id='c::s',case_id='c',dataset_id='d',dataset_version='1',participant_id='p',session_id='s',task_segment_id='t',device_id='dev',split='train',stroke_id='s',point_refs=('p1',),mapping_status='MAPPED',mapping_method='ARC_LENGTH',episode_index=0,quality_status='VALID')
    events=tuple(SyntheticTruthEvent(event_id=f'e{i}',event_type=label,source_point_ids=('p1',)) for i,label in enumerate(LABEL_ORDER))
    labels,records=align_truth((episode,),{'c':events})
    vector=target_vector(labels[episode.episode_id])
    assert vector == (1,1,1,1,1,1,1,1)
    assert all(record.alignment_status=='ONE_TO_ONE' for record in records)


def test_formula_helpers():
    assert _iou((0,0,1,1),(0.5,0.5,1.5,1.5)) == pytest.approx(1/7)
    assert _question_history_value({},None,'UNKNOWN') == (None,None)
    assert _duration_s(1000, 1500) == pytest.approx(0.5)
    assert _duration_s(None, 1500) is None
    assert _pause_before_s(2500, 1500) == pytest.approx(1.0)
    assert _pause_before_s(None, 1500) is None
    assert _pause_inside_s([0, 400, 1100], 500) == pytest.approx(0.2)
    assert _pause_inside_s([0, None], 500) is None
    class P:
        def __init__(self,x,y): self.x_norm=x; self.y_norm=y
    assert _norm_path([P(0,0),P(0.3,0.4)], ('a','b')) == pytest.approx(0.5)
    assert _norm_path([P(0,0),P(None,0.4)], ('a','b')) is None


def test_feature_dataset_round_trip_uses_tmp_path(tmp_path):
    config=load_dataset_config('configs/r1_synthetic_dataset_v1.json').model_copy(update={'participant_count':3})
    source=tmp_path/'source'; materialize_dataset(config,source)
    output=tmp_path/'features'; result=materialize_features(source,output,overwrite=False)
    ds=reload_feature_dataset(output)
    assert result.manifest.manifest_hash == ds['manifest'].manifest_hash
    assert len(ds['alignment_records']) == ds['summary']['alignment_audit']['truth_event_count']
    assert ds['split_manifests'][0].source_dataset_manifest_hash == ds['manifest'].source_dataset_manifest_hash


def test_canonical_contracts_match_runtime_artifacts(tmp_path):
    config=load_dataset_config('configs/r1_synthetic_dataset_v1.json').model_copy(update={'participant_count':3})
    source=tmp_path/'source'; materialize_dataset(config,source)
    output=tmp_path/'features'; materialize_features(source,output)
    feature,target=_canonical_contracts()
    assert json.loads((output/'feature_schema.json').read_text()) == feature
    assert json.loads((output/'target_spec.json').read_text()) == target


def test_json_schemas_validate_assets_and_reject_invalid_rows(tmp_path):
    config=load_dataset_config('configs/r1_synthetic_dataset_v1.json').model_copy(update={'participant_count':3})
    source=tmp_path/'source'; materialize_dataset(config,source)
    output=tmp_path/'features'; materialize_features(source,output)
    for filename,schema_file in [('episodes.jsonl','r1_episode_schema_v1.json'),('feature_rows.jsonl','r1_feature_row_schema_v1.json'),('sequence_samples.jsonl','r1_sequence_sample_schema_v1.json')]:
        schema=json.loads(Path('contracts',schema_file).read_text())
        validator=Draft202012Validator(schema)
        for line in (output/filename).read_text().splitlines(): validator.validate(json.loads(line))
    row=json.loads((output/'feature_rows.jsonl').read_text().splitlines()[0]); row['features']['scenario_type']=1
    with pytest.raises(Exception): Draft202012Validator(json.loads(Path('contracts/r1_feature_row_schema_v1.json').read_text())).validate(row)
