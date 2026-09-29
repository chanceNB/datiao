from datiao.r1.features.audit import run_feature_leakage_audit
from datiao.r1.features.models import FeatureRow, FEATURE_ORDER

def test_leakage_audit_rejects_extra_feature():
    row=FeatureRow.model_construct(episode_id='e',split='train',features={**{k:0.0 for k in FEATURE_ORDER},'participant_id':1.0},target=(0,)*8)
    audit=run_feature_leakage_audit((row,))
    assert audit['status']=='FAIL'
    assert 'participant_id' in audit['outside_whitelist']
