"""Compile the supplied taxonomy into a ragged tree with L2/L3 leaves."""
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
        'matching_mode': 'leaf_only',
        'hint_semantics': '第三列按用户要求作为三级节点；有三级则仅匹配三级，否则匹配二级本身',
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


def validate(config):
    unhashed = {k: v for k, v in config.items() if k != 'fingerprint'}
    if config['fingerprint'] != digest(unhashed):
        raise ValueError('Taxonomy fingerprint mismatch')
    ps = config['primary']
    ss = [s for p in ps for s in p['secondaries']]
    if not ps or not ss or any(not p['secondaries'] for p in ps):
        raise ValueError('Taxonomy requires nonempty primary and secondary categories')
    children = [c for s in ss for c in s['children']]
    if config.get('schema_version') != 2 or config.get('matching_mode') != 'leaf_only':
        raise ValueError('Expected leaf-only schema; regenerate the taxonomy')
    ids = [x['id'] for x in ps + ss + children + config['production_forms']]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate category ID')
    if len({p['name'] for p in ps}) != len(ps):
        raise ValueError('Duplicate primary name')
    for p in ps:
        if len({s['name'] for s in p['secondaries']}) != len(p['secondaries']):
            raise ValueError(f'Duplicate secondary in {p["name"]}')
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
    """One full-path vector for each leaf; ancestors never compete with children."""
    if separator not in (' / ', ' '):
        raise ValueError('Unsupported path separator')
    rows = []
    for p in config['primary']:
        for s in p['secondaries']:
            for leaf in s['children'] or [s]:
                path = [p['name'], s['name']] + ([leaf['name']] if s['children'] else [])
                rows.append({'id': leaf['id'], 'name': leaf['name'],
                             'primary_id': p['id'], 'secondary_id': s['id'],
                             'tertiary_id': leaf['id'] if s['children'] else None,
                             'level': 3 if s['children'] else 2, 'path': path,
                             'text': separator.join(path)})
    return rows


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
