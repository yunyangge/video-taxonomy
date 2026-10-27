"""Versioned taxonomy centers, tag lookup, streaming video classification and shares."""
import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
from .taxonomy import ROOT, centers, digest, load
from .core import UNKNOWN, choose_leaf, clean, duration_seconds, eligibility, evidence, form_result, video_key


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        tmp = Path(f.name)
    tmp.replace(path)


def run_config(args, create=False):
    config = load(args.config)
    path = args.run_dir / 'run.json'
    stored = json.loads(path.read_text()) if path.exists() else None
    requested = getattr(args, 'path_separator', None)
    separator = {'slash': ' / ', 'space': ' '}.get(requested, stored.get('path_separator', ' / ') if stored else ' / ')
    spec = {'taxonomy_fingerprint': config['fingerprint'],
            'embedding': config['embedding'], 'centers': centers(config, separator)}
    if separator != ' / ':
        spec['path_separator'] = separator
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != spec:
            raise ValueError('Run directory contains another taxonomy/embedding; use a new directory')
    elif create:
        atomic_json(path, spec)
    else:
        raise ValueError('Run embed-centers first in a new run directory')
    return config, spec


class Encoder:
    def __init__(self, model, device, spec):
        import torch
        from transformers import AutoModel, AutoTokenizer
        if not model.is_dir():
            raise ValueError(f'Model directory not found: {model}')
        self.torch, self.device, self.spec = torch, device, spec
        self.tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, padding_side='left')
        dtype = torch.bfloat16 if device.startswith('cuda') else torch.float32
        self.model = AutoModel.from_pretrained(model, local_files_only=True, dtype=dtype,
                                               attn_implementation='sdpa').to(device).eval()

    def encode(self, texts):
        import numpy as np
        if not texts:
            return np.empty((0, self.spec['dimensions']), dtype=np.float32)
        with self.torch.inference_mode():
            tokens = self.tokenizer([clean(x) for x in texts], padding=True, truncation=True,
                                    max_length=self.spec['max_length'], return_tensors='pt').to(self.device)
            x = self.model(**tokens).last_hidden_state[:, -1, :].float()
            if x.shape[1] != self.spec['dimensions']:
                raise ValueError('Embedding dimension mismatch')
            return self.torch.nn.functional.normalize(x, dim=1).cpu().numpy()


class Matcher:
    def __init__(self, args, config, spec):
        import numpy as np
        self.np, self.args = np, args
        self.ids = [leaf['id'] for leaf in spec['centers']]
        path = args.run_dir / 'center_vectors.npy'
        meta_path = args.run_dir / 'center_vectors.json'
        meta = json.loads(meta_path.read_text())
        if meta['run_fingerprint'] != digest(spec) or hashlib.sha256(path.read_bytes()).hexdigest() != meta['sha256']:
            raise ValueError('Stale or changed center vectors')
        self.matrix = np.load(path).astype(np.float32)
        if self.matrix.shape != (len(spec['centers']), config['embedding']['dimensions']):
            raise ValueError('Center shape mismatch')
        if not np.isfinite(self.matrix).all() or (np.linalg.norm(self.matrix, axis=1) == 0).any():
            raise ValueError('Invalid centers')
        self.matrix /= np.linalg.norm(self.matrix, axis=1, keepdims=True)

    def match(self, vectors):
        np = self.np
        x = np.asarray(vectors, dtype=np.float32)
        valid = np.isfinite(x).all(axis=1) & (np.linalg.norm(x, axis=1) > 0)
        x = np.nan_to_num(x)
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        similarity = x @ self.matrix.T
        scores = similarity
        best = scores.argmax(axis=1)
        top = scores[np.arange(len(scores)), best]
        second = np.partition(scores, -2, axis=1)[:, -2]
        margin = top - second
        accepted = valid & (top >= self.args.min_score) & (margin >= self.args.min_margin)
        return np.where(accepted, best + 1, 0).astype(np.uint16), top, margin


def embed_centers(args):
    import numpy as np
    config, spec = run_config(args, create=True)
    out = args.run_dir / 'center_vectors.npy'
    if out.exists():
        raise FileExistsError('Centers already exist; use a new run directory')
    encoder = Encoder(args.model, args.device, config['embedding'])
    rows = spec['centers']
    matrix = np.concatenate([encoder.encode([r['text'] for r in rows[i:i + args.batch_size]])
                             for i in range(0, len(rows), args.batch_size)])
    tmp = out.with_suffix('.tmp.npy')
    np.save(tmp, matrix.astype(np.float16))
    tmp.replace(out)
    atomic_json(out.with_suffix('.json'), {'run_fingerprint': digest(spec),
                'sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
                'model_name': args.model.name, 'embedding': config['embedding']})
    print(json.dumps({'center_count': len(rows), 'dimensions': matrix.shape[1], 'output': str(out)}))


def assign_tags(args):
    """Reuse only the documented full-1024 cache; classify against NEW centers."""
    import numpy as np
    import pyarrow.parquet as pq
    config, spec = run_config(args)
    matcher = Matcher(args, config, spec)
    source = pq.ParquetFile(args.tag_source)
    cache = json.loads((args.legacy_path_vectors / 'center_encoding.json').read_text())
    if cache['dimensions'] != 1024 or Path(cache['model']).name != config['embedding']['model_name']:
        raise ValueError('Legacy vector model/dimensions are incompatible')
    # This adapter is specifically for path-centers-1024/embed_full.py (NFC,
    # 2048 chars, 96 tokens, last-token pooling, L2, zeroed old noise rows).
    folder = args.run_dir / 'tag_lookup'
    folder.mkdir(exist_ok=True)
    if (folder / 'metadata.json').exists():
        raise FileExistsError('Tag lookup exists; use another run directory for new parameters')
    count = source.metadata.num_rows
    raw = {name: np.lib.format.open_memmap(folder / f'{name}_raw.npy', mode='w+', dtype=dtype, shape=(count,))
           for name, dtype in [('h1', '<u8'), ('h2', '<u8'), ('labels', 'uint16'), ('scores', 'float32'), ('margins', 'float32')]}
    offset = 0
    for g in range(source.num_row_groups):
        tags = source.read_row_group(g, columns=['tag'])['tag'].to_pylist()
        x = np.load(args.legacy_path_vectors / 'vectors' / f'{g:06d}.npy', mmap_mode='r')
        if x.shape != (len(tags), 1024):
            raise ValueError(f'Invalid legacy shard {g}')
        hashes = np.frombuffer(b''.join(hashlib.md5(t.encode()).digest() for t in tags), dtype='<u8').reshape(-1, 2)
        stop = offset + len(tags)
        raw['h1'][offset:stop], raw['h2'][offset:stop] = hashes[:, 0], hashes[:, 1]
        for start in range(0, len(tags), args.batch_size):
            labels, scores, margins = matcher.match(x[start:start + args.batch_size])
            sl = slice(offset + start, offset + start + len(labels))
            raw['labels'][sl], raw['scores'][sl], raw['margins'][sl] = labels, scores, margins
        offset = stop
        print(json.dumps({'tag_shard': g, 'rows_done': offset}), flush=True)
    if offset != count:
        raise ValueError('Tag row count mismatch')
    order = np.argsort(raw['h1'], kind='stable')
    for name, arr in raw.items():
        arr.flush()
        dest = np.lib.format.open_memmap(folder / f'{name}.npy', mode='w+', dtype=arr.dtype, shape=arr.shape)
        for start in range(0, count, 1000000):
            dest[start:start + 1000000] = arr[order[start:start + 1000000]]
        dest.flush()
    atomic_json(folder / 'metadata.json', {'taxonomy_fingerprint': config['fingerprint'],
                'run_fingerprint': digest(spec), 'rows': count, 'leaf_ids': matcher.ids,
                'min_score': args.min_score, 'min_margin': args.min_margin,
                'source': args.tag_source.name,
                'source_bytes': args.tag_source.stat().st_size,
                'legacy_path_vectors': args.legacy_path_vectors.name,
                'inherited_old_noise_mask': True})


class TagLookup:
    def __init__(self, path, args, config, spec):
        import numpy as np
        self.np = np
        metadata = json.loads((path / 'metadata.json').read_text())
        if metadata['run_fingerprint'] != digest(spec) or metadata['taxonomy_fingerprint'] != config['fingerprint']:
            raise ValueError('Tag lookup taxonomy mismatch')
        if (metadata['min_score'], metadata['min_margin']) != (args.min_score, args.min_margin):
            raise ValueError('Tag lookup thresholds mismatch')
        self.arrays = {name: np.load(path / f'{name}.npy', mmap_mode='r') for name in ['h1', 'h2', 'labels', 'scores', 'margins']}
        if any(len(x) != metadata['rows'] for x in self.arrays.values()):
            raise ValueError('Incomplete tag lookup')
        self.ids = metadata['leaf_ids']
        if self.ids != [leaf['id'] for leaf in spec['centers']]:
            raise ValueError('Tag lookup leaf order mismatch')

    def get(self, text):
        np, a = self.np, self.arrays
        h1, h2 = np.frombuffer(hashlib.md5(text.encode()).digest(), dtype='<u8')
        lo, hi = np.searchsorted(a['h1'], h1, side='left'), np.searchsorted(a['h1'], h1, side='right')
        found = np.flatnonzero(a['h2'][lo:hi] == h2)
        if not len(found):
            return None
        i = lo + found[0]
        code = int(a['labels'][i])
        return (self.ids[code - 1] if code else UNKNOWN, float(a['scores'][i]), float(a['margins'][i]))


def input_batches(args):
    def batches(path, groups=None):
        if path.suffix == '.parquet':
            import pyarrow.parquet as pq
            for batch in pq.ParquetFile(path).iter_batches(batch_size=args.batch_size, row_groups=groups):
                yield batch.to_pylist()
        else:
            if groups is not None:
                raise ValueError('Row groups require Parquet input')
            with path.open() as f:
                rows = []
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
                    if len(rows) == args.batch_size:
                        yield rows
                        rows = []
                if rows:
                    yield rows
    if args.tasks:
        for task in json.loads(args.tasks.read_text()):
            # Legacy non-BVD tasks are already separate per-row-group files.
            groups = task.get('groups', [task['group']]) if task['corpus'] == 'bvd' else None
            for batch in batches(Path(task['path']), groups):
                yield task['corpus'], batch
    else:
        for path in args.input:
            for batch in batches(path):
                yield args.platform, batch


def classify(args):
    config, spec = run_config(args)
    matcher = Matcher(args, config, spec)
    lookup = TagLookup(args.tag_lookup, args, config, spec) if args.tag_lookup else None
    secondary = {s['id']: s for p in config['primary'] for s in p['secondaries']}
    parents = {s['id']: p['id'] for p in config['primary'] for s in p['secondaries']}
    leaves = {leaf['id']: leaf for leaf in spec['centers']}
    encoder = None
    if args.output.exists():
        raise FileExistsError('Classification output already exists')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + '.partial')
    if tmp.exists():
        raise FileExistsError(f'Incomplete previous output: {tmp}; use a new output filename')
    # Disk-based exact dedup spans every file/task in this invocation.
    with tempfile.TemporaryDirectory(prefix='video-dedup-', dir=args.output.parent) as temp:
        seen = sqlite3.connect(str(Path(temp) / 'ids.sqlite'))
        seen.execute('CREATE TABLE seen (key TEXT PRIMARY KEY) WITHOUT ROWID')
        stats = Counter()
        with tmp.open('w') as out:
            for default_platform, batch in input_batches(args):
                unique = []
                for row in batch:
                    stats['source_rows'] += 1
                    platform = str(row.get('platform') or default_platform)
                    key = video_key(row, platform)
                    if key is not None:
                        if seen.execute('INSERT OR IGNORE INTO seen VALUES (?)', (key,)).rowcount == 0:
                            stats['duplicate_rows'] += 1
                            continue
                    else:
                        stats['rows_without_identity'] += 1
                    unique.append((platform, key, row, evidence(row)))
                texts = list(dict.fromkeys(t for _, _, _, ev in unique for t in ev))
                predictions = {}
                unresolved = []
                for text in texts:
                    found = lookup.get(text) if lookup else None
                    if found is not None:
                        predictions[text] = found
                    else:
                        unresolved.append(text)
                if unresolved:
                    if args.lookup_only:
                        predictions.update({x: (UNKNOWN, 0.0, 0.0) for x in unresolved})
                        stats['unmatched_evidence'] += len(unresolved)
                    else:
                        if encoder is None:
                            encoder = Encoder(args.model, args.device, config['embedding'])
                        for start in range(0, len(unresolved), args.batch_size):
                            part = unresolved[start:start + args.batch_size]
                            labels, scores, margins = matcher.match(encoder.encode(part))
                            for text, label, score, margin in zip(part, labels, scores, margins):
                                predictions[text] = (matcher.ids[label - 1] if label else UNKNOWN, float(score), float(margin))
                for platform, key, row, ev in unique:
                    votes = [(predictions[t][0], predictions[t][1]) for t in ev]
                    pid, sid, lid = choose_leaf(votes, spec['centers'])
                    status = 'metadata_candidate' if sid != UNKNOWN else 'insufficient_evidence'
                    override = row.get('content_secondary_id')
                    if override is not None:
                        if override != UNKNOWN and override not in secondary:
                            raise ValueError(f'Invalid content_secondary_id: {override}')
                        sid, pid = override, parents.get(override, UNKNOWN)
                        lid = sid if sid in leaves else UNKNOWN
                        status = 'provided'
                    leaf_override = row.get('content_leaf_id')
                    if leaf_override is not None:
                        if leaf_override != UNKNOWN and leaf_override not in leaves:
                            raise ValueError(f'Invalid content_leaf_id: {leaf_override}')
                        leaf = leaves.get(leaf_override)
                        if override is not None and override != (leaf['secondary_id'] if leaf else UNKNOWN):
                            raise ValueError('Supplied leaf and secondary disagree')
                        lid = leaf_override
                        pid, sid = (leaf['primary_id'], leaf['secondary_id']) if leaf else (UNKNOWN, UNKNOWN)
                        status = 'provided'
                    form, form_status, form_candidates = form_result(row, config)
                    duration, duration_kind = duration_seconds(row)
                    result = {'taxonomy_version': config['taxonomy_version'], 'taxonomy_fingerprint': config['fingerprint'],
                              'platform': platform, 'video_key': key,
                              'video_id': row.get('video_id', row.get('id')), 'url': row.get('url', row.get('webpage_url')),
                              'primary_id': pid, 'secondary_id': sid, 'content_status': status,
                              'leaf_id': lid, 'leaf_level': leaves[lid]['level'] if lid in leaves else None,
                              'tertiary_id': leaves[lid]['tertiary_id'] if lid in leaves else None,
                              'production_form_id': form, 'form_status': form_status,
                              'form_candidates': form_candidates, 'review_required': status != 'provided' or form_status != 'provided',
                              'duration_seconds': duration, 'duration_kind': duration_kind,
                              'eligibility': eligibility(row, secondary.get(sid)),
                              'evidence_count': len(ev), 'accepted_votes': sum(l in leaves for l, _ in votes),
                              'title': row.get('title')}
                    out.write(json.dumps(result, ensure_ascii=False) + '\n')
                    stats['unique_rows'] += 1
                    stats['unclassified_content'] += sid == UNKNOWN
                seen.commit()
        seen.close()
    metadata = {'taxonomy_fingerprint': config['fingerprint'], 'run_fingerprint': digest(spec),
                'min_score': args.min_score, 'min_margin': args.min_margin,
                'stats': dict(stats), 'tasks': args.tasks.name if args.tasks else None,
                'inputs': [p.name for p in args.input], 'completed': True,
                'lookup_only': args.lookup_only, 'semantic_boundary_status': 'metadata_candidates_require_review'}
    # Completion marker is committed last; aggregate refuses interrupted files.
    tmp.replace(args.output)
    atomic_json(args.output.with_suffix(args.output.suffix + '.meta.json'), metadata)
    print(json.dumps(metadata, ensure_ascii=False))


def aggregate(args):
    from .core import quota_status
    config = load(args.config)
    if args.output.exists():
        raise FileExistsError('Aggregate output exists; use a new filename')
    prim = {p['id']: p for p in config['primary']}
    second = {s['id']: s for p in config['primary'] for s in p['secondaries']}
    parents = {s['id']: p['id'] for p in config['primary'] for s in p['secondaries']}
    forms = {f['id']: f for f in config['production_forms']}
    leaves = {leaf['id']: dict(leaf, quota={}) for leaf in centers(config)}
    counts, seconds = Counter(), Counter()
    totals, effective = Counter(), Counter()
    duration_coverage, eligibility_counts = Counter(), Counter()
    input_records = 0
    common_run_fingerprint = None
    with tempfile.TemporaryDirectory(prefix='aggregate-dedup-', dir=args.output.parent if args.output.parent.exists() else None) as temp:
        db = sqlite3.connect(str(Path(temp) / 'seen.sqlite'))
        db.execute('CREATE TABLE seen (key TEXT PRIMARY KEY) WITHOUT ROWID')
        for path in args.input:
            meta = json.loads(path.with_suffix(path.suffix + '.meta.json').read_text())
            if meta['taxonomy_fingerprint'] != config['fingerprint'] or not meta['completed']:
                raise ValueError('Incomplete or incompatible classification output')
            if common_run_fingerprint is None:
                common_run_fingerprint = meta['run_fingerprint']
            elif meta['run_fingerprint'] != common_run_fingerprint:
                raise ValueError('Different center representations cannot be aggregated together')
            rows = 0
            with path.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    rows += 1
                    if row['taxonomy_fingerprint'] != config['fingerprint']:
                        raise ValueError('Mixed taxonomy rows')
                    key = row['video_key']
                    if key is not None and db.execute('INSERT OR IGNORE INTO seen VALUES (?)', (key,)).rowcount == 0:
                        raise ValueError('Duplicate video across result files; classify inputs together')
                    pid, sid, fid = row['primary_id'], row['secondary_id'], row['production_form_id']
                    if (pid == UNKNOWN) != (sid == UNKNOWN) or (sid != UNKNOWN and parents.get(sid) != pid) or fid not in {*forms, UNKNOWN}:
                        raise ValueError('Invalid content hierarchy/form')
                    lid = row['leaf_id']
                    if lid != UNKNOWN and (lid not in leaves or leaves[lid]['secondary_id'] != sid or leaves[lid]['primary_id'] != pid):
                        raise ValueError('Invalid leaf hierarchy')
                    if row['tertiary_id'] != (leaves[lid]['tertiary_id'] if lid in leaves else None):
                        raise ValueError('Invalid tertiary ID')
                    platform = row['platform']
                    eligibility_counts[(platform, row['eligibility'])] += 1
                    # Optional procurement audit uses only explicit human-approved rows.
                    if args.accepted_only and (row['eligibility'] != 'not_excluded_by_metadata' or row['review_required']):
                        continue
                    totals[platform] += 1
                    duration = row['duration_seconds']
                    valid_duration = duration is not None and row['duration_kind'] == 'effective'
                    if valid_duration:
                        effective[platform] += duration
                        duration_coverage[platform] += 1
                    for dimension, cat in [('primary', pid), ('secondary', sid), ('leaf', lid), ('production_form', fid)]:
                        counts[(platform, dimension, cat)] += 1
                        if valid_duration:
                            seconds[(platform, dimension, cat)] += duration
            if rows != meta['stats']['unique_rows']:
                raise ValueError('Classification output row count changed')
            input_records += rows
        db.close()
    rows = []
    for platform, total in totals.items():
        ready = args.accepted_only and duration_coverage[platform] == total and effective[platform] > 0
        for dimension, categories in [('primary', prim), ('secondary', second), ('leaf', leaves), ('production_form', forms)]:
            classified = total - counts[(platform, dimension, UNKNOWN)]
            for cat in [*categories, UNKNOWN]:
                key = (platform, dimension, cat)
                count, duration = counts[key], seconds[key]
                item = categories.get(cat, {'name': '无法归类', 'quota': {}})
                share = duration / effective[platform] if effective[platform] else None
                parent_count = counts[(platform, 'primary', parents[cat])] if dimension == 'secondary' and cat in parents else None
                if dimension == 'leaf' and cat in leaves:
                    parent_count = counts[(platform, 'secondary', leaves[cat]['secondary_id'])] if leaves[cat]['level'] == 3 else counts[(platform, 'primary', leaves[cat]['primary_id'])]
                rows.append({'platform': platform, 'dimension': dimension, 'id': cat, 'name': item['name'],
                             'parent_id': parents.get(cat) if dimension == 'secondary' else (leaves[cat]['secondary_id'] if dimension == 'leaf' and cat in leaves and leaves[cat]['level'] == 3 else (leaves[cat]['primary_id'] if dimension == 'leaf' and cat in leaves else None)),
                             'leaf_level': leaves[cat]['level'] if dimension == 'leaf' and cat in leaves else None,
                             'video_count': count, 'share_all': count / total,
                             'share_classified': count / classified if cat != UNKNOWN and classified else None,
                             'share_parent': count / parent_count if parent_count else None,
                             'effective_duration_seconds': duration,
                             'share_effective_duration': share,
                             'quota': item['quota'],
                             'quota_status': quota_status(share or 0, duration / 3600, item['quota'], ready) if cat != UNKNOWN and dimension != 'leaf' else 'not_applicable'})
        for dimension in ['primary', 'secondary', 'leaf', 'production_form']:
            if sum(r['video_count'] for r in rows if r['platform'] == platform and r['dimension'] == dimension) != total:
                raise ValueError('Category counts do not partition videos')
    atomic_json(args.output, {'taxonomy_version': config['taxonomy_version'], 'taxonomy_fingerprint': config['fingerprint'],
                'input_records': input_records, 'included_videos': dict(totals),
                'accepted_only': args.accepted_only, 'effective_duration_rows': dict(duration_coverage),
                'effective_duration_totals': dict(effective),
                'eligibility_counts': [{'platform': p, 'status': status, 'count': n} for (p, status), n in eligibility_counts.items()],
                'quota_note': '只有 accepted-only 且有效时长完整才评估配额；候选分布不代表采购验收。无时长不以条数代替。',
                'rows': rows})
    print(json.dumps({'output': str(args.output), 'included_videos': dict(totals)}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=os.environ.get('VIDEO_TAXONOMY_CONFIG'), help='External taxonomy JSON; no taxonomy data is bundled')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ['embed-centers', 'assign-tags', 'classify']:
        p = sub.add_parser(name)
        p.add_argument('--run-dir', type=Path, required=True)
        p.add_argument('--model', type=Path, default=Path(os.environ.get('VIDEO_MODEL_DIR', 'models/Qwen3-Embedding-0.6B')))
        p.add_argument('--device', default='cpu')
        p.add_argument('--batch-size', type=int, default=128)
        p.add_argument('--min-score', type=float, default=0.0)
        p.add_argument('--min-margin', type=float, default=0.0)
        p.add_argument('--path-separator', choices=['slash', 'space'], default=None,
                       help='Center path joiner; defaults to the saved run or slash for a new run')
    p = sub.choices['assign-tags']
    p.add_argument('--legacy-path-vectors', type=Path, required=True)
    p.add_argument('--tag-source', type=Path, required=True)
    p = sub.choices['classify']
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--input', type=Path, nargs='+', default=[])
    source.add_argument('--tasks', type=Path)
    p.add_argument('--platform', default='unspecified')
    p.add_argument('--tag-lookup', type=Path)
    p.add_argument('--lookup-only', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('aggregate')
    p.add_argument('--input', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--accepted-only', action='store_true')
    args = parser.parse_args()
    if not args.config:
        parser.error('Provide --config or VIDEO_TAXONOMY_CONFIG; no taxonomy data is bundled')
    if getattr(args, 'batch_size', 1) < 1:
        parser.error('--batch-size must be positive')
    if getattr(args, 'lookup_only', False) and not args.tag_lookup:
        parser.error('--lookup-only requires --tag-lookup')
    {'embed-centers': embed_centers, 'assign-tags': assign_tags, 'classify': classify, 'aggregate': aggregate}[args.command](args)


if __name__ == '__main__':
    main()
