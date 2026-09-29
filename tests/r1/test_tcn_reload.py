import inspect

from datiao.r1.tcn.io import _per_label_deltas, reload_tcn_run
from datiao.r1.features.models import LABEL_ORDER


def test_reload_requires_frozen_feature_dataset_path():
    params = inspect.signature(reload_tcn_run).parameters
    assert "features" in params
    assert params["features"].default is inspect.Parameter.empty


def test_per_label_deltas_use_canonical_label_order():
    def metric(value):
        return {"per_label": {label: {"f1": value, "roc_auc": None} for label in LABEL_ORDER}}

    result = _per_label_deltas(metric(1.0), metric(0.25))
    assert list(result) == list(LABEL_ORDER)
    assert result["QUESTION_LEAVE"]["f1_delta"] == 0.75
