from pathlib import Path
from datiao.r1.features.models import FEATURE_ORDER, LABEL_ORDER, FeatureRow, SequenceSample
from datiao.r1.features.io import reload_feature_dataset

def test_feature_order_target_and_mask_contract():
    row=FeatureRow(episode_id='e', split='train', features={k:(None if k=='path_length_norm' else 1.0) for k in FEATURE_ORDER}, target=(1,0,0,0,0,0,0,0))
    assert tuple(row.features)==FEATURE_ORDER
    sample=SequenceSample(sequence_id='s',case_id='c',participant_id='p',task_segment_id='t',split='train',episode_ids=('e',),values=((0.0,)*12,),feature_mask=((0,1,1,1,1,1,1,1,1,1,1,1),),targets=(row.target,))
    assert sample.feature_mask[0][1]==1 and sample.feature_mask[0][0]==0

def test_feature_dataset_round_trip():
    path=Path('artifacts/r1_synthetic_features_v1')
    if not path.exists(): return
    ds=reload_feature_dataset(path)
    assert ds['manifest'].episode_count==700
    assert ds['manifest'].sequence_count==360
    assert ds['leakage_audit']['status']=='PASS'
