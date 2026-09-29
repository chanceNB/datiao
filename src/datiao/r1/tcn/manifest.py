"""Canonical semantic identities separate from artifact container bytes."""
import hashlib
import json
from ..lightgbm.manifest import sha256_file, write_json, write_jsonl


def semantic_json_hash(value):
    encoded = json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')
    return 'sha256:' + hashlib.sha256(encoded).hexdigest()


def semantic_run_hash(manifest):
    excluded = {'run_hash','manifest_integrity_hash','model_file_hash','environment','wall_time','timestamp','absolute_path','hostname','username'}
    return semantic_json_hash({k:v for k,v in manifest.items() if k not in excluded})


def manifest_integrity_hash(manifest):
    return semantic_json_hash({k:v for k,v in manifest.items() if k!='manifest_integrity_hash'})
