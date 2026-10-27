"""Pure classification and accounting rules; no model or filesystem side effects."""
import math
import re
import unicodedata
from collections import Counter
from urllib.parse import parse_qs, urlsplit

UNKNOWN = 'unclassified'


def clean(value):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', str(value or ''))).strip()[:2048]


def useful(value):
    text = clean(value)
    if not text or text.lower() in {'null', 'none', 'untitled', 'unknown', 'video'}:
        return False
    if '\ufffd' in text or re.fullmatch(r'[\W\d_]+', text):
        return False
    if re.fullmatch(r'(?:https?://\S+|(?:vid(?:eo)?|img|dsc|clip)[_ -]*\d+)', text, re.I):
        return False
    return True


def evidence(row, source='tags'):
    tags = row.get('tags') or []
    if isinstance(tags, str):
        tags = tags.split(',')
    if source not in {'tags', 'metadata'}:
        raise ValueError('Unknown evidence source')
    values = ([row.get('title'), row.get('description')] if source == 'metadata' else []) + list(tags)
    # As in the previous vote pipeline, one repeated string is one vote.
    return list(dict.fromkeys(clean(x) for x in values if useful(x)))


def video_key(row, platform):
    url = str(row.get('webpage_url') or row.get('url') or '')
    identifier = row.get('video_id')
    if identifier is None or str(identifier).strip() == '':
        identifier = row.get('id')
    if identifier is not None and str(identifier).strip():
        if platform == 'bvd':
            host = (urlsplit(url).hostname or '').removeprefix('www.')
            if host in {'youtu.be', 'm.youtube.com'}:
                host = 'youtube.com'
            # Unknown hosts must not make IDs collide across suppliers.
            return f'{platform}:{host}:{identifier}' if host else (f'{platform}:url:{url}' if url else None)
        return f'{platform}:id:{identifier}'
    if url:
        parsed = urlsplit(url)
        host = (parsed.hostname or '').removeprefix('www.')
        if host in {'youtube.com', 'm.youtube.com', 'youtu.be'}:
            key = (parsed.path.strip('/') if host == 'youtu.be' else parse_qs(parsed.query).get('v', [None])[0])
            if key:
                return f'{platform}:youtube.com:{key}'
        return f'{platform}:url:{url}'
    return None


def choose(votes, primary_by_secondary, secondary_order):
    """L1 majority, then L2 within that L1; ties by score sum then source order."""
    valid = [(sid, float(score)) for sid, score in votes if sid in primary_by_secondary and math.isfinite(float(score))]
    if not valid:
        return UNKNOWN, UNKNOWN
    counts = Counter(primary_by_secondary[sid] for sid, _ in valid)
    sums = Counter()
    for sid, score in valid:
        sums[primary_by_secondary[sid]] += score
    pid = min(counts, key=lambda x: (-counts[x], -sums[x], x))
    selected = [(sid, score) for sid, score in valid if primary_by_secondary[sid] == pid]
    counts = Counter(sid for sid, _ in selected)
    sums = Counter()
    for sid, score in selected:
        sums[sid] += score
    sid = min(counts, key=lambda x: (-counts[x], -sums[x], secondary_order[x]))
    return pid, sid


def choose_leaf(votes, leaves):
    """Vote L1, then L2 inside the winner, then leaf inside winning L2."""
    mapping = {leaf['id']: leaf for leaf in leaves}
    valid = [(lid, float(score)) for lid, score in votes
             if lid in mapping and math.isfinite(float(score))]
    parents = {leaf['secondary_id']: leaf['primary_id'] for leaf in leaves}
    order = {leaf['secondary_id']: i for i, leaf in reversed(list(enumerate(leaves)))}
    pid, sid = choose([(mapping[lid]['secondary_id'], score) for lid, score in valid], parents, order)
    if sid == UNKNOWN:
        return UNKNOWN, UNKNOWN, UNKNOWN
    eligible = [(lid, score) for lid, score in valid if mapping[lid]['secondary_id'] == sid]
    counts, sums = Counter(lid for lid, _ in eligible), Counter()
    leaf_order = {leaf['id']: i for i, leaf in enumerate(leaves)}
    for lid, score in eligible:
        sums[lid] += score
    lid = min(counts, key=lambda x: (-counts[x], -sums[x], leaf_order[x]))
    return pid, sid, lid


def choose_path(votes, nodes):
    """Vote L1 first; direct ancestor votes may stop descent at L1 or L2."""
    mapping = {node['id']: node for node in nodes}
    order = {node['id']: i for i, node in enumerate(nodes)}
    valid = [(nid, float(score)) for nid, score in votes
             if nid in mapping and math.isfinite(float(score))]
    if not valid:
        return UNKNOWN, UNKNOWN, UNKNOWN

    def winner(entries):
        counts, sums = Counter(), Counter()
        for nid, score in entries:
            counts[nid] += 1
            sums[nid] += score
        return min(counts, key=lambda nid: (-counts[nid], -sums[nid], order[nid]))

    pid = winner([(mapping[nid]['primary_id'], score) for nid, score in valid])
    selected = [(nid, score) for nid, score in valid if mapping[nid]['primary_id'] == pid]
    # The parent itself is a real endpoint; do not discard its votes or
    # invent a child when those votes win against the child branches.
    sid = winner([(mapping[nid]['secondary_id'] or pid, score) for nid, score in selected])
    if sid == pid:
        return pid, UNKNOWN, pid
    nid = winner([(nid, score) for nid, score in selected if mapping[nid]['secondary_id'] == sid])
    return pid, sid, nid


# Only explicit metadata signals are used for form candidates. Broad words such
# as "game", "animation" or "music" would confuse the topic with the format.
FORM_PATTERNS = {
    '专业影视与节目': r'电影片段|电视剧片段|影视剪辑|电影解说|电视剧解说|综艺节目|纪录片|赛事转播|电影预告|\b(?:movie clip|film trailer|documentary|tv show|sports broadcast)\b',
    '广告与商业设计': r'广告片|宣传片|品牌片|企业形象片|\b(?:TVC|brand film|commercial ad)\b',
    '音乐与舞蹈影像': r'官方MV|音乐MV|舞蹈视频|舞蹈翻跳|演唱会录像|乐器演奏视频|\b(?:music video|dance cover|concert recording)\b',
    '动画与 CG': r'动画短片|动画片|三维动画|动漫片段|\b(?:animated film|animated short|CG animation)\b',
    '游戏画面': r'游戏实况|游戏录屏|电竞比赛|游戏解说|\b(?:gameplay|playthrough|lets? play|walkthrough game)\b',
    '屏幕录制与图文': r'屏幕录制|录屏教程|PPT讲解|图文轮播|\b(?:screen recording|screencast|slideshow)\b',
    '实拍记录': r'\bvlog\b|行车记录仪画面|实拍教程|生活记录|实拍素材',
}


def form_result(row, config):
    lookup = {f['id']: f['id'] for f in config['production_forms']}
    lookup.update({f['name']: f['id'] for f in config['production_forms']})
    supplied = row.get('production_forms') or row.get('production_form') or []
    if isinstance(supplied, str):
        supplied = [supplied]
    if any(x not in lookup for x in supplied):
        raise ValueError(f'Unknown supplied production form: {supplied}')
    candidates = {lookup[x] for x in supplied}
    if supplied:
        status = 'provided'
    else:
        # Hashtags often name a topic; they do not establish production form.
        text = clean(row.get('title')) + ' ' + clean(row.get('description'))
        candidates = {lookup[name] for name, pattern in FORM_PATTERNS.items() if name in lookup and re.search(pattern, text, re.I)}
        status = 'metadata_candidate' if candidates else 'insufficient_evidence'
    selected = next((x for x in config['production_form_priority'] if x in candidates), UNKNOWN)
    return selected, status, sorted(candidates)


def duration_seconds(row):
    value = row.get('effective_duration_seconds')
    kind = 'effective'
    if value is None:
        value = row.get('duration_seconds')
        kind = 'raw'
    if value is None:
        return None, 'missing'
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None, 'invalid'
    return (value, kind) if math.isfinite(value) and value > 0 else (None, 'invalid')


def eligibility(row, secondary):
    value = row.get('is_aigc')
    # Do not interpret string "false" as Python's truthy True.
    if isinstance(value, str):
        if value.lower() in {'true', '1', 'yes'}:
            value = True
        elif value.lower() in {'false', '0', 'no'}:
            value = False
        else:
            raise ValueError('is_aigc must be a boolean or true/false string')
    if value is not None and value not in (True, False):
        raise ValueError('is_aigc must be boolean')
    if value is True:
        return 'excluded_aigc'
    if secondary and secondary['restricted']:
        return 'restricted_pending_review'
    return 'aigc_not_checked' if value is None else 'not_excluded_by_metadata'


def quota_status(actual_share, actual_hours, quota, ready):
    if not ready:
        return 'not_evaluable'
    if quota.get('min_share') is not None and actual_share < quota['min_share']:
        return 'below_minimum'
    if quota.get('max_share') is not None and actual_share > quota['max_share']:
        return 'above_maximum'
    if quota.get('min_hours') is not None and actual_hours < quota['min_hours']:
        return 'below_minimum'
    return 'within_range'
