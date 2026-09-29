from __future__ import annotations
from typing import Any
import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, hamming_loss,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)
from ..features.models import LABEL_ORDER

def _single_metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    pred=(probability>=threshold).astype(np.int8)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel().tolist()
    single=len(np.unique(y))<2
    return {"support_positive": int(y.sum()), "support_negative": int(len(y)-y.sum()),
            "precision": float(precision_score(y,pred,zero_division=0)), "recall": float(recall_score(y,pred,zero_division=0)),
            "f1": float(f1_score(y,pred,zero_division=0)), "mcc": float(matthews_corrcoef(y,pred)),
            "roc_auc": None if single else float(roc_auc_score(y,probability)),
            "average_precision": None if single else float(average_precision_score(y,probability)),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            "metric_status": "SINGLE_CLASS_UNDEFINED" if single else "OK"}

def calculate_metrics(Y: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions=(probabilities>=threshold).astype(np.int8)
    per_label={label:_single_metrics(Y[:,i],probabilities[:,i],threshold) for i,label in enumerate(LABEL_ORDER)}
    aucs=[v["roc_auc"] for v in per_label.values() if v["roc_auc"] is not None]
    aps=[v["average_precision"] for v in per_label.values() if v["average_precision"] is not None]
    return {"threshold": threshold, "per_label": per_label,
            "aggregate":{"micro_precision":float(precision_score(Y.ravel(),predictions.ravel(),zero_division=0)),
                "micro_recall":float(recall_score(Y.ravel(),predictions.ravel(),zero_division=0)),
                "micro_f1":float(f1_score(Y.ravel(),predictions.ravel(),zero_division=0)),
                "macro_f1_all_labels":float(f1_score(Y,predictions,average="macro",zero_division=0)),
                "macro_roc_auc_evaluable_labels":None if not aucs else float(np.mean(aucs)),
                "macro_average_precision_evaluable_labels":None if not aps else float(np.mean(aps)),
                "evaluable_roc_auc_labels":[label for label,v in per_label.items() if v["roc_auc"] is not None],
                "evaluable_average_precision_labels":[label for label,v in per_label.items() if v["average_precision"] is not None],
                "hamming_loss":float(hamming_loss(Y,predictions)),
                "subset_accuracy":float(np.mean(np.all(Y==predictions,axis=1)))}}
