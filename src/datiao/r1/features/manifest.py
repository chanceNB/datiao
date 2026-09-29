from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any
from ..models.immutable import stable_json_hash

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return f"sha256:{h.hexdigest()}"

def compute_feature_manifest_hash(value: Any) -> str:
    payload = value.model_dump(mode='json') if hasattr(value,'model_dump') else dict(value)
    payload.pop('manifest_hash', None)
    return f"sha256:{stable_json_hash(payload)}"

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',',':'), allow_nan=False)+'\n', encoding='utf-8', newline='\n')

def write_jsonl(path: Path, rows: list[dict[str,Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False) for x in rows)+'\n', encoding='utf-8', newline='\n')

def read_json(path: Path) -> dict[str,Any]:
    value=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value,dict): raise ValueError(f'expected object: {path}')
    return value

def read_jsonl(path: Path) -> list[dict[str,Any]]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
