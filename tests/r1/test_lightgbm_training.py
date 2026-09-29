import numpy as np
import pytest
from datiao.r1.lightgbm.model import ConstantBinaryPredictor, LightGBMBinaryPredictor, fit_predictor
from datiao.r1.lightgbm.metrics import calculate_metrics
from datiao.r1.lightgbm.config import LightGBMConfig
from datiao.r1.lightgbm.data import input_integrity_audit, FeatureMatrices
from datiao.r1.features.models import FEATURE_ORDER, LABEL_ORDER

def test_missing_matrix_semantics_and_shape():
    X=np.array([[0.0,np.nan]+[0.0]*10],dtype=float); Y=np.zeros((1,8),dtype=np.int8)
    data=FeatureMatrices(dataset={'manifest':type('M',(),{})()},X={'train':X,'validation':X,'test':X},Y={'train':Y,'validation':Y,'test':Y},episode_ids={'train':('a',),'validation':('b',),'test':('c',)})
    audit=input_integrity_audit(data)
    assert audit['status']=='PASS'; assert X[0,0]==0.0 and np.isnan(X[0,1])

def test_constant_predictor_round_trip(tmp_path):
    predictor=ConstantBinaryPredictor('WRITING',1); path=tmp_path/'WRITING.constant.json'; predictor.save(path); loaded=ConstantBinaryPredictor.load(path)
    assert loaded.predict_proba(np.zeros((3,12))).tolist()==[1.0,1.0,1.0]
    assert loaded.learner_type=='CONSTANT_POSITIVE'

def test_trainable_label_fit_save_reload_and_deterministic_probability(tmp_path):
    X=np.array([[0.0],[0.1],[0.9],[1.0],[0.2],[0.8]],dtype=float); y=np.array([0,0,1,1,0,1],dtype=np.int8)
    config=LightGBMConfig(num_boost_round=20,early_stopping_rounds=5,lightgbm_params={**LightGBMConfig().lightgbm_params,'num_leaves':3,'min_data_in_leaf':1})
    path=tmp_path/'LABEL.txt'; predictor=fit_predictor('LABEL',X,y,X,y,config,path); loaded=LightGBMBinaryPredictor.load(path,'LABEL',predictor.best_iteration)
    assert predictor.learner_type=='LIGHTGBM'; assert np.allclose(predictor.predict_proba(X),loaded.predict_proba(X),atol=1e-12)

def test_single_class_metrics_are_undefined_for_auc_and_ap():
    result=calculate_metrics(np.ones((3,8),dtype=np.int8),np.ones((3,8),dtype=float),0.5)
    assert result['per_label']['WRITING']['roc_auc'] is None
    assert result['per_label']['WRITING']['average_precision'] is None
    assert result['per_label']['WRITING']['metric_status']=='SINGLE_CLASS_UNDEFINED'
