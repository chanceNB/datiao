import json
import numpy as np
import pytest

from datiao.r1.tcn.config import TCNConfig, load_tcn_config
from datiao.r1.evaluation.multilabel import calculate_metrics, calculate_subset_metrics
from datiao.r1.lightgbm.metrics import calculate_metrics as legacy
from datiao.r1.lightgbm.manifest import write_json, sha256_file


def test_official_config_is_fixed_and_rejects_tuning():
    config = load_tcn_config('configs/r1_tcn_v1.json')
    assert config.hidden_channels == (32, 32)
    assert config.device == 'cpu'
    assert config.threshold == .5
    with pytest.raises(ValueError):
        TCNConfig(hidden_channels=(64, 64))


def test_shared_metrics_preserve_frozen_lightgbm_json(tmp_path):
    rows = [json.loads(line) for line in open('artifacts/r1_lightgbm_v1/predictions.jsonl', encoding='utf-8')]
    metrics = {split: calculate_metrics(np.array([r['target'] for r in rows if r['split']==split]), np.array([r['probability_vector'] for r in rows if r['split']==split]), .5) for split in ('validation','test')}
    write_json(tmp_path/'metrics.json', metrics)
    assert sha256_file(tmp_path/'metrics.json') == 'sha256:440ebc9995e70b5035b88c2513d5284301ba3026413999ae688cb1cb5c0e0fdc'
    assert legacy is calculate_metrics


def test_subset_metrics_use_only_named_labels():
    y = np.array([[1,0,1,0,0,0,0,0],[1,1,0,0,0,0,0,0]])
    result = calculate_subset_metrics(y, y.astype(float), .5, ('WRITING','QUESTION_VISIT'))
    assert list(result['per_label']) == ['WRITING','QUESTION_VISIT']
    assert result['aggregate']['micro_f1'] == 1.
