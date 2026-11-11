"""Verified seed-2000 inference bundles and matching distance profiles."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def catalog(root=ROOT):
    root = Path(root).resolve()
    value = json.loads((root/'03_checkpoints/manifest.json').read_text(encoding='utf-8'))
    if value.get('version') != 1 or value.get('seed') != 2000:
        raise ValueError('Unsupported checkpoint catalog')
    entries = value['checkpoints']
    if len({row['id'] for row in entries}) != len(entries):
        raise ValueError('Duplicate checkpoint identifiers')
    return entries


def checked_path(root, relative):
    root = Path(root).resolve()
    path = (root/relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Checkpoint path escapes the checkout')
    return path


def verify(entry, root=ROOT):
    files = {}
    for role in ('asset', 'router'):
        if role not in entry:
            continue
        path = checked_path(root, entry[role])
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry[role+'_sha256']:
            raise ValueError('Checkpoint checksum mismatch: '+entry['id']+'/'+role)
        files[role] = json.loads(data)
    asset = files['asset']
    if entry['distance'] == 'hausdorff' and entry['target'] == 'pointset':
        if asset.get('supervision_distance') != 'SDD point-set Hausdorff':
            raise ValueError('Point-set checkpoint has another training target')
    if 'router' in files:
        for key in ('geometry_mean', 'geometry_std'):
            if asset[key] != files['router'][key]:
                raise ValueError('DTW probes and router have different normalization')
    return files


def profile_name(entry, mode='batch'):
    if mode not in ('batch','single','auto'):
        raise ValueError('mode must be batch, single or auto')
    suffix = 'pointset_hd' if entry['distance']=='hausdorff' and entry['target']=='pointset' else entry['distance']
    return ('auto_' if mode=='auto' else mode+'_query_')+suffix+'.json'


def resolve(identifier, root=ROOT, mode='batch'):
    rows = [row for row in catalog(root) if row['id'] == identifier]
    if len(rows) != 1:
        raise ValueError('Unknown checkpoint: '+identifier)
    row = dict(rows[0])
    verify(row,root)
    row['asset_path'] = checked_path(root,row['asset'])
    row['profile_path'] = checked_path(root,'02_configs/'+row['distance']+'/'+profile_name(row,mode))
    row['router_path'] = checked_path(root,row['router']) if 'router' in row else None
    if not row['profile_path'].is_file():
        raise FileNotFoundError(row['profile_path'])
    return row


def load_checkpoint(identifier, device='cuda:0', *, mode='batch', root=ROOT):
    entry = resolve(identifier,root,mode)
    from .common.config import RetrievalConfig
    from .common.dual_field import DualFieldRetriever
    return DualFieldRetriever.from_asset(entry['asset_path'],device=device,
        config=RetrievalConfig.from_json(entry['profile_path']),routing_asset=entry['router_path'])
