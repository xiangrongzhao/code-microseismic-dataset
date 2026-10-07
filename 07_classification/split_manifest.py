"""Generate and verify within-class chronological record or whole-day splits."""
from pathlib import Path
from datetime import datetime
import argparse
import hashlib
import json
import re
import pandas as pd

CLASSES = ['microseismic', 'blasting', 'small_landslide', 'mechanical_mining', 'transport_vehicles']
PHASES = ['train', 'val', 'test']
POLICIES = ['record', 'whole-day-next']
PATTERN = re.compile(r'^sensor_(\d{9})_(\d{8}T\d{6}\.\d{6})_(\d{8}T\d{6}\.\d{6})\.npy$')

def allocation(n):
    weights = [7, 2, 1]
    sizes = [n*w//10 for w in weights]
    order = sorted(range(3), key=lambda i: (-(n*weights[i] % 10), i))
    for i in order[:n-sum(sizes)]:
        sizes[i] += 1
    return sizes

def parse_name(path):
    match = PATTERN.fullmatch(path.name)
    if not match:
        raise ValueError(f'Invalid event filename: {path.name}')
    start = datetime.strptime(match[2], '%Y%m%dT%H%M%S.%f')
    end = datetime.strptime(match[3], '%Y%m%dT%H%M%S.%f')
    if end < start:
        raise ValueError(f'End before start: {path.name}')
    return match[1], start.isoformat(timespec='microseconds'), end.isoformat(timespec='microseconds')

def split_counts(start_times, policy='record'):
    """Keep the original 70%/90% cuts; move a split date entirely forward."""
    if policy not in POLICIES:
        raise ValueError(f'Unknown split policy: {policy}')
    sizes = allocation(len(start_times))
    if policy == 'record':
        return sizes
    cuts = [sizes[0], sizes[0]+sizes[1]]
    dates = [str(t)[:10] for t in start_times]
    for i, cut in enumerate(cuts):
        if 0 < cut < len(dates) and dates[cut-1] == dates[cut]:
            date = dates[cut]
            while cut > 0 and dates[cut-1] == date:
                cut -= 1
            cuts[i] = cut
    return [cuts[0], cuts[1]-cuts[0], len(dates)-cuts[1]]


def make_manifest(root, policy='record'):
    root = Path(root).resolve()
    rows = []
    for idx, cls in enumerate(CLASSES):
        candidates = []
        for p in (root/cls).glob('*.npy'):
            sensor, start, end = parse_name(p)
            candidates.append({'relative_path': p.relative_to(root).as_posix(), 'class_dir': cls,
                'class_key': cls, 'class_idx': idx, 'sensor_id': sensor,
                'start_time': start, 'end_time': end,
                'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
        candidates.sort(key=lambda r: (r['start_time'], r['end_time'], r['sensor_id'], r['relative_path']))
        cursor = 0
        for phase, n in zip(PHASES, split_counts([r['start_time'] for r in candidates], policy)):
            rows.extend([{**r, 'split': phase, 'split_policy': policy} for r in candidates[cursor:cursor+n]])
            cursor += n
    return pd.DataFrame(rows)

def load_manifest(root, path):
    root = Path(root).resolve()
    df = pd.read_csv(path, dtype={'sensor_id': str, 'sha256': str})
    if 'split_policy' not in df.columns:
        df['split_policy'] = 'record'
    if df.split_policy.isna().any() or df.split_policy.nunique() != 1 or df.split_policy.iloc[0] not in POLICIES:
        raise ValueError('Manifest must contain one recognized split policy.')
    policy = df.split_policy.iloc[0]
    required = {'relative_path', 'class_dir', 'class_key', 'class_idx', 'sensor_id',
                'start_time', 'end_time', 'split', 'sha256'}
    if not required.issubset(df.columns) or df.empty or df[list(required)].isna().any().any():
        raise ValueError('Missing manifest fields or values.')
    if df.relative_path.duplicated().any() or not set(df.split).issubset(PHASES):
        raise ValueError('Duplicate file assignments or invalid subset names.')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*.npy')}
    if actual != set(df.relative_path):
        raise ValueError('Manifest file inventory does not match the dataset.')
    absolute = []
    for r in df.itertuples():
        p = (root/r.relative_path).resolve()
        if not p.is_relative_to(root) or not p.is_file():
            raise ValueError(f'Unsafe or missing input path: {r.relative_path}')
        sensor, start, end = parse_name(p)
        if (sensor, start, end) != (r.sensor_id, r.start_time, r.end_time):
            raise ValueError(f'Manifest timestamp/ID mismatch: {r.relative_path}')
        if r.class_key not in CLASSES or r.class_idx != CLASSES.index(r.class_key) or r.class_dir != r.class_key or p.parent.name != r.class_key:
            raise ValueError(f'Manifest label mismatch: {r.relative_path}')
        if hashlib.sha256(p.read_bytes()).hexdigest() != r.sha256:
            raise ValueError(f'Waveform hash mismatch: {r.relative_path}')
        absolute.append(str(p))
    for cls in CLASSES:
        group = df[df.class_key == cls].sort_values(['start_time', 'end_time', 'sensor_id', 'relative_path'])
        expected = [phase for phase,n in zip(PHASES, split_counts(group.start_time.tolist(), policy)) for _ in range(n)]
        if group.split.tolist() != expected or any((group.split == p).sum() == 0 for p in PHASES):
            raise ValueError(f'Non-chronological, empty, or incorrect {policy} allocation: {cls}')
        if policy == 'whole-day-next' and group.groupby(group.start_time.str[:10]).split.nunique().max() != 1:
            raise ValueError(f'A UTC date is shared between subsets within {cls}.')
    if df.sha256.duplicated().any():
        raise ValueError('Byte-identical input files remain; resolve before training.')
    df.insert(0, 'path', absolute)
    return df

def assert_cache_signature(old, current):
    if old != current:
        raise RuntimeError('Cache/run signature differs: use a new --out-dir. Dataset, split, settings, code, and versions must match.')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--split-policy',choices=POLICIES,default='record')
    args=p.parse_args()
    if args.out.resolve().is_relative_to(args.data_root.resolve()):
        p.error('Save the manifest outside the waveform directory.')
    frame=make_manifest(args.data_root,args.split_policy)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(args.out,index=False)
    checked=load_manifest(args.data_root,args.out)
    print(checked.groupby(['class_key','split']).size().unstack().to_string())
    print(json.dumps({'files':len(checked),'manifest_sha256':hashlib.sha256(args.out.read_bytes()).hexdigest()}))

if __name__=='__main__':
    main()
