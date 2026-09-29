from __future__ import annotations
import json
from pathlib import Path
from .manifest import run_hash, sha256_file
from ..features.io import reload_feature_dataset
from ..features.models import FEATURE_ORDER, LABEL_ORDER

def reload_lightgbm_run(output: str|Path, features: str|Path|None = None):
    root=Path(output); manifest=json.loads((root/'run_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('run_hash') != run_hash(manifest): raise ValueError('run manifest hash mismatch')
    if tuple(manifest.get('feature_order',())) != FEATURE_ORDER or tuple(manifest.get('label_order',())) != LABEL_ORDER: raise ValueError('run frozen order mismatch')
    if features is not None:
        feature_manifest=reload_feature_dataset(features)['manifest']
        if manifest['source_feature_manifest_hash'] != feature_manifest.manifest_hash: raise ValueError('run feature lineage mismatch')
    model_dir=root/'models'
    for name,expected in manifest['model_file_hashes'].items():
        path=model_dir/name
        if not path.exists() or sha256_file(path)!=expected: raise ValueError(f'model hash mismatch: {name}')
    if sha256_file(root/'predictions.jsonl') != manifest['predictions_hash']: raise ValueError('predictions hash mismatch')
    if sha256_file(root/'metrics.json') != manifest['metrics_hash']: raise ValueError('metrics hash mismatch')
    return {'output_path':str(root),'manifest':manifest,'metrics':json.loads((root/'metrics.json').read_text(encoding='utf-8')),'predictions_hash':manifest['predictions_hash'],'metrics_hash':manifest['metrics_hash']}
