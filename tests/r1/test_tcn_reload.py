import inspect
import json
import shutil

import pytest

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


def test_reload_checks_both_run_and_manifest_integrity_hash(tmp_path):
    output = tmp_path / "tcn"
    shutil.copytree("artifacts/r1_tcn_v1", output)
    manifest_path = output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["manifest_integrity_hash"] = "sha256:tampered"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest integrity hash mismatch"):
        reload_tcn_run(output, "artifacts/r1_synthetic_features_v1")
