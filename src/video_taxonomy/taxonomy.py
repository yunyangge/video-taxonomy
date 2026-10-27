"""Compile external taxonomies into all-path centers (scheme 3)."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def quota(text):
    if '限制' in text or '不设' in text or text == '—':
        return {'min_share': None, 'max_share': None, 'min_hours': None}
    percentages = [float(x) / 100 for x in re.findall(r'(\d+(?:\.\d+)?)%', text)]
    hours = re.search(r'([\d,]+)\s*小时', text)
    return {'min_share': percentages[0] if percentages else None,
            'max_share': percentages[1] if len(percentages) > 1 else None,
            'min_hours': float(hours[1].replace(',', '')) if hours else None}


def compile_markdown(path):
    raw = Path(path).read_bytes()
    text = raw.decode('utf-8')
    section = ''
    primary = []
    forms = []
    rules = []
    restrictions = {}
    current = None
    limits = {}
    for line in text.splitlines():
        if line.startswith('## '):
            section = line
        if section.startswith('## 一、') and re.match(r'^\d+\.', line):
            rules.append(re.sub(r'^\d+\.\s*', '', line))
        heading = re.match(r'^### (\d+)\. (.+)$', line)
        if section.startswith('## 四、') and heading:
            current = {'id': f'C{int(heading[1]):02d}', 'name': heading[2],
                       'secondaries': []}
            primary.append(current)
        if not line.startswith('|'):
            continue
        cells = [c.strip() for c in line.strip('|').split('|')]
        if not cells or cells[0].startswith('---'):
            continue
        if section.startswith('## 三、') and cells[0] != '一级类目':
            limits[cells[0]] = {'min_share': float(cells[1].rstrip('%')) / 100,
                               'max_share': float(cells[2].rstrip('%')) / 100,
                               'note': cells[3]}
        if section.startswith('## 四、') and current and cells[0] != '二级类目':
            name, tier, hints = cells
            current['secondaries'].append({
                'id': f'{current["id"]}-{len(current["secondaries"]) + 1:02d}',
                'name': name, 'tier': tier, 'quota': quota(tier),
                'restricted': tier.startswith('限制'),
                'children': [] if hints == '—' else [
                    {'id': f'{current["id"]}-{len(current["secondaries"]) + 1:02d}-{i:02d}',
                     'name': name.strip()}
                    for i, name in enumerate(hints.split('；'), 1) if name.strip()]})
        if section.startswith('## 五、') and cells[0] != '制作形态':
            forms.append({'id': f'F{len(forms) + 1:02d}', 'name': cells[0],
                          'quota_text': cells[1], 'quota': quota(cells[1]),
                          'definition': cells[2]})
        if section.startswith('## 六、') and cells[0] != '类目':
            restrictions[cells[0]] = {'handling': cells[1], 'reason': cells[2]}
    for p in primary:
        p['quota'] = limits[p['name']]
        for s in p['secondaries']:
            s['restriction'] = restrictions.get(p['name'] + '·' + s['name'])
    by_form = {f['name']: f['id'] for f in forms}
    match = re.search(r'多种体裁并存时按此顺序取一个：(.+)。', text)
    priority = [by_form[x.strip()] for x in match[1].split('>')]
    result = {
        'schema_version': 2, 'taxonomy_version': 'custom',
        'source': {'filename': Path(path).name, 'sha256': hashlib.sha256(raw).hexdigest()},
        'levels': ['primary', 'secondary', 'tertiary'],
        'matching_mode': 'all_paths',
        'hint_semantics': '一级、一级到二级、一级到末端路径共同参与匹配；二级叶子只计一次',
        'rules': rules, 'primary': primary, 'production_forms': forms,
        'production_form_priority': priority, 'restrictions': restrictions,
        'unknown_id': 'unclassified',
        'embedding': {'model_name': 'Qwen3-Embedding-0.6B', 'dimensions': 1024,
                      'pooling': 'last_token_left_padding', 'max_length': 96,
                      'normalization': 'NFC_whitespace_2048chars_L2'},
        'quota_basis': 'effective_duration_seconds',
    }
    result['fingerprint'] = digest(result)
    validate(result)
    return result


def validate(config, allow_legacy=False):
    def require_fields(value, names, location):
        if not isinstance(value, dict):
            raise ValueError(f'{location} must be an object')
        missing = [name for name in names if name not in value]
        if missing:
            raise ValueError(f'{location} missing required fields: {", ".join(missing)}')

    require_fields(config, ['fingerprint', 'taxonomy_version', 'schema_version', 'matching_mode',
                            'primary', 'production_forms', 'production_form_priority',
                            'embedding', 'unknown_id'], 'Configuration')
    require_fields(config['embedding'], ['model_name', 'dimensions', 'pooling', 'max_length',
                                         'normalization'], 'embedding')
    if config['unknown_id'] != 'unclassified':
        raise ValueError('unknown_id must be unclassified')
    for p in config['primary']:
        require_fields(p, ['id', 'name', 'quota', 'secondaries'], 'Primary category')
        for s in p['secondaries']:
            require_fields(s, ['id', 'name', 'quota', 'restricted', 'children'], 'Secondary category')
            for c in s['children']:
                require_fields(c, ['id', 'name'], 'Tertiary category')
    for f in config['production_forms']:
        require_fields(f, ['id', 'name', 'quota'], 'Production form')
    unhashed = {k: v for k, v in config.items() if k != 'fingerprint'}
    if config['fingerprint'] != digest(unhashed):
        raise ValueError('Taxonomy fingerprint mismatch')
    ps = config['primary']
    ss = [s for p in ps for s in p['secondaries']]
    if not ps or not ss or any(not p['secondaries'] for p in ps):
        raise ValueError('Taxonomy requires nonempty primary and secondary categories')
    children = [c for s in ss for c in s['children']]
    modes = {'all_paths', 'leaf_only'} if allow_legacy else {'all_paths'}
    if config.get('schema_version') != 2 or config.get('matching_mode') not in modes:
        raise ValueError('Expected matching_mode=all_paths (scheme 3); use migrate-config for a legacy config')
    if config['embedding']['dimensions'] != 1024:
        raise ValueError('Scheme 3 requires full 1024-dimensional embeddings')
    ids = [x['id'] for x in ps + ss + children + config['production_forms']]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate category ID')
    if len({p['name'] for p in ps}) != len(ps):
        raise ValueError('Duplicate primary name')
    for p in ps:
        if len({s['name'] for s in p['secondaries']}) != len(p['secondaries']):
            raise ValueError(f'Duplicate secondary in {p["name"]}')
        for s in p['secondaries']:
            if len({c['name'] for c in s['children']}) != len(s['children']):
                raise ValueError(f'Duplicate child in {s["name"]}')
    if set(config['production_form_priority']) != {f['id'] for f in config['production_forms']} or len(config['production_form_priority']) != len(config['production_forms']):
        raise ValueError('Invalid form priority')


def load(path=None):
    selected = path or os.environ.get('VIDEO_TAXONOMY_CONFIG')
    if not selected:
        raise ValueError('Provide --config or VIDEO_TAXONOMY_CONFIG; no private taxonomy is bundled')
    config = json.loads(Path(selected).read_text())
    validate(config)
    return config


def centers(config, separator=' / '):
    """L1 + L1/L2 + L1/leaf; emit each node once in taxonomy order."""
    if separator not in (' / ', ' '):
        raise ValueError('Unsupported path separator')
    rows = []
    for p in config['primary']:
        rows.append({'id': p['id'], 'name': p['name'], 'primary_id': p['id'],
                     'secondary_id': None, 'tertiary_id': None, 'level': 1,
                     'is_leaf': False, 'path': [p['name']], 'text': p['name']})
        for s in p['secondaries']:
            path = [p['name'], s['name']]
            rows.append({'id': s['id'], 'name': s['name'],
                         'primary_id': p['id'], 'secondary_id': s['id'],
                         'tertiary_id': None, 'level': 2, 'is_leaf': not s['children'],
                         'path': path, 'text': separator.join(path)})
            for leaf in s['children']:
                path = [p['name'], s['name'], leaf['name']]
                rows.append({'id': leaf['id'], 'name': leaf['name'],
                             'primary_id': p['id'], 'secondary_id': s['id'],
                             'tertiary_id': leaf['id'], 'level': 3, 'is_leaf': True, 'path': path,
                             'text': separator.join(path)})
    return rows


def migrate_config(config):
    """Change matching policy without importing or changing category definitions."""
    validate(config, allow_legacy=True)
    result = json.loads(json.dumps(config))
    result['matching_mode'] = 'all_paths'
    result['hint_semantics'] = '一级、一级到二级、一级到末端路径共同参与匹配；二级叶子只计一次'
    result.pop('fingerprint', None)
    result['fingerprint'] = digest(result)
    validate(result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('markdown', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = compile_markdown(args.markdown)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'primary': len(config['primary']), 'secondary': sum(len(p['secondaries']) for p in config['primary']), 'centers': len(centers(config)),
                      'fingerprint': config['fingerprint']}, ensure_ascii=False))
