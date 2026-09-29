from __future__ import annotations

import json
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field

class LightGBMConfig(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    baseline_id: str = "r1-lightgbm-ovr-v1"
    baseline_version: str = "1.0.0"
    seed: int = 20260929
    threshold: float = Field(default=0.5, ge=0, le=1)
    num_boost_round: int = Field(default=200, ge=1)
    early_stopping_rounds: int = Field(default=20, ge=1)
    lightgbm_params: dict[str, object] = {
        "objective": "binary", "metric": "binary_logloss", "learning_rate": 0.05,
        "num_leaves": 15, "min_data_in_leaf": 10, "feature_fraction": 1.0,
        "bagging_fraction": 1.0, "lambda_l2": 1.0, "verbosity": -1,
        "deterministic": True, "num_threads": 1,
    }

def load_config(path: str | Path) -> LightGBMConfig:
    return LightGBMConfig.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
