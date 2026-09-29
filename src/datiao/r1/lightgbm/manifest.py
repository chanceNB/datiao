from __future__ import annotations
import hashlib, json
from pathlib import Path
from ..models.immutable import stable_json_hash

def sha256_file(path):
    h=hashlib.sha256();
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return f"sha256:{h.hexdigest()}"
def write_json(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True); Path(path).write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n',encoding='utf-8')
def write_jsonl(path,rows):
    Path(path).parent.mkdir(parents=True,exist_ok=True); Path(path).write_text('\n'.join(json.dumps(r,ensure_ascii=False,sort_keys=True,separators=(',',':')) for r in rows)+'\n',encoding='utf-8')
def run_hash(manifest):
    payload=dict(manifest); payload.pop('run_hash',None); return f"sha256:{stable_json_hash(payload)}"
