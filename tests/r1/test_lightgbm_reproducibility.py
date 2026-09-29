import inspect, json
from pathlib import Path
import numpy as np
import pytest
from datiao.r1.lightgbm.io import reload_lightgbm_run
from datiao.r1.lightgbm.manifest import run_hash, sha256_file, write_json
from datiao.r1.lightgbm.metrics import calculate_metrics
from datiao.r1.lightgbm.model import fit_predictor
from datiao.r1.features.models import FEATURE_ORDER, LABEL_ORDER

def test_metrics_are_independent_of_prediction_serialization_order():
    y=np.zeros((3, len(LABEL_ORDER)),dtype=np.int8); y[:, :2]=np.array([[0,1],[1,0],[1,1]],dtype=np.int8)
    p=np.full((3, len(LABEL_ORDER)),.1,dtype=float); p[:, :2]=np.array([[.1,.8],[.9,.2],[.7,.6]],dtype=float)
    first=calculate_metrics(y,p,.5)
    ids=('e3','e1','e2'); rows=[{'episode_id':ids[i],'target':y[i].tolist(),'probability_vector':p[i].tolist()} for i in range(3)]
    ordered=sorted(rows,key=lambda row:row['episode_id']); by_id={row['episode_id']:row for row in ordered}
    reconstructed=np.asarray([by_id[eid]['probability_vector'] for eid in ids]); targets=np.asarray([by_id[eid]['target'] for eid in ids])
    assert calculate_metrics(targets,reconstructed,.5)==first

def test_fit_api_has_no_test_inputs():
    names=set(inspect.signature(fit_predictor).parameters)
    assert 'X_test' not in names and 'y_test' not in names

def test_run_manifest_self_hash_and_tamper_detection(tmp_path):
    root=tmp_path/'run'; (root/'models').mkdir(parents=True)
    model=root/'models'/'WRITING.constant.json'; model.write_text('{"constant_class":1}\n',encoding='utf-8')
    (root/'predictions.jsonl').write_text('{}\n',encoding='utf-8'); (root/'metrics.json').write_text('{}\n',encoding='utf-8')
    manifest={'run_version':'1','feature_order':list(FEATURE_ORDER),'label_order':list(LABEL_ORDER),'model_file_hashes':{model.name:sha256_file(model)},'predictions_hash':sha256_file(root/'predictions.jsonl'),'metrics_hash':sha256_file(root/'metrics.json')}
    manifest['run_hash']=run_hash(manifest); write_json(root/'run_manifest.json',manifest)
    assert reload_lightgbm_run(root)['manifest']['run_hash']==manifest['run_hash']
    model_bytes=model.read_bytes(); model.write_text('{"tampered":true}\n',encoding='utf-8')
    with pytest.raises(ValueError,match='model hash mismatch'): reload_lightgbm_run(root)
    model.write_bytes(model_bytes)
    prediction_bytes=(root/'predictions.jsonl').read_bytes(); (root/'predictions.jsonl').write_text('{"tampered":true}\n',encoding='utf-8')
    with pytest.raises(ValueError,match='predictions hash mismatch'): reload_lightgbm_run(root)
    (root/'predictions.jsonl').write_bytes(prediction_bytes)
    (root/'metrics.json').write_text('{"tampered":true}\n',encoding='utf-8')
    with pytest.raises(ValueError,match='metrics hash mismatch'): reload_lightgbm_run(root)

def test_repro_audit_records_independent_hash_comparisons():
    path=Path('artifacts/r1_lightgbm_v1/audits/reproducibility.json')
    if not path.exists(): pytest.fail('full baseline artifact was not materialized')
    audit=json.loads(path.read_text(encoding='utf-8'))
    assert audit['same_prediction_hash'] is True
    assert audit['same_metrics_hash'] is True
    assert audit['same_run_hash'] is True
    assert audit['run1_prediction_hash']==audit['run2_prediction_hash']
    assert audit['run1_metrics_hash']==audit['run2_metrics_hash']
    assert audit['run1_run_hash']==audit['run2_run_hash']
