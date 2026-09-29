import pytest
from datiao.r1.features.audit import run_feature_leakage_audit
from datiao.r1.features.models import FEATURE_ORDER, FeatureRow

@pytest.mark.parametrize('bad_key', ['scenario_type','participant_id','truth_event_type','predicted_event_type','split'])
def test_leakage_inputs_are_rejected_by_normal_validation(bad_key):
    features={key:0.0 for key in FEATURE_ORDER}
    if bad_key == 'split':
        with pytest.raises(ValueError): FeatureRow(episode_id='e',split='train',features={**features,bad_key:1.0},target=(0,)*8)
    else:
        with pytest.raises(ValueError): FeatureRow(episode_id='e',split='train',features={**features,bad_key:1.0},target=(0,)*8)

def test_leakage_audit_remains_independent_guard():
    row=FeatureRow(episode_id='e',split='train',features={key:0.0 for key in FEATURE_ORDER},target=(0,)*8)
    audit=run_feature_leakage_audit((row,))
    assert audit['status']=='PASS'
