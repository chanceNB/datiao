from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import lightgbm as lgb
from .config import LightGBMConfig
from ..features.models import FEATURE_ORDER

class ConstantBinaryPredictor:
    def __init__(self, label_name: str, constant_class: int): self.label_name=label_name; self.constant_class=int(constant_class)
    @property
    def learner_type(self): return "CONSTANT_POSITIVE" if self.constant_class==1 else "CONSTANT_NEGATIVE"
    @property
    def best_iteration(self): return 0
    def predict_proba(self, X):
        p=np.full(len(X),float(self.constant_class)); return p
    def save(self,path:Path): path.write_text(json.dumps({"label_name":self.label_name,"constant_class":self.constant_class,"learner_type":self.learner_type},sort_keys=True,separators=(',',':'))+'\n',encoding='utf-8')
    @classmethod
    def load(cls,path):
        value=json.loads(Path(path).read_text(encoding='utf-8')); return cls(value['label_name'],value['constant_class'])

class LightGBMBinaryPredictor:
    def __init__(self, label_name: str, booster: lgb.Booster, best_iteration: int, feature_order=FEATURE_ORDER): self.label_name=label_name; self.booster=booster; self._best_iteration=best_iteration; self.feature_order=tuple(feature_order)
    @property
    def learner_type(self): return "LIGHTGBM"
    @property
    def best_iteration(self): return self._best_iteration
    def predict_proba(self,X): return np.asarray(self.booster.predict(X,num_iteration=self._best_iteration),dtype=np.float64)
    def save(self,path:Path): path.write_text(self.booster.model_to_string(num_iteration=self._best_iteration), encoding="utf-8", newline="\n")
    @classmethod
    def load(cls,path,label_name,best_iteration): return cls(label_name,lgb.Booster(model_str=Path(path).read_text(encoding="utf-8")),best_iteration)

def fit_predictor(label_name: str, X_train, y_train, X_validation, y_validation, config: LightGBMConfig, path: Path | None):
    classes=np.unique(y_train)
    if len(classes)<2:
        predictor=ConstantBinaryPredictor(label_name,int(classes[0]));
        if path: predictor.save(path)
        return predictor
    params=dict(config.lightgbm_params); params.update({"random_state":config.seed,"feature_fraction_seed":config.seed,"bagging_seed":config.seed,"data_random_seed":config.seed})
    estimator=lgb.LGBMClassifier(n_estimators=config.num_boost_round,**params)
    estimator.fit(X_train,y_train,eval_set=[(X_validation,y_validation)],callbacks=[lgb.early_stopping(config.early_stopping_rounds,verbose=False)])
    best=int(estimator.best_iteration_ or config.num_boost_round)
    predictor=LightGBMBinaryPredictor(label_name,estimator.booster_,best)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        predictor.save(path)
    return predictor
