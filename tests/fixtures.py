"""Generated test-only taxonomy. No real category definitions or business data."""
from video_taxonomy.core import FORM_PATTERNS
from video_taxonomy.taxonomy import digest


def sample_config():
    unlimited = {'min_share': None, 'max_share': None, 'min_hours': None}
    def secondary(code, children=0, restricted=False):
        return {'id': code, 'name': 'Example ' + code, 'tier': 'example',
                'quota': dict(unlimited), 'restricted': restricted, 'restriction': None,
                'children': [{'id': f'{code}-{i:03}', 'name': f'Example leaf {i}'} for i in range(children)]}
    forms = [{'id': f'F{i:02}', 'name': name, 'definition': '', 'quota': dict(unlimited), 'quota_text': ''}
             for i, name in enumerate(FORM_PATTERNS, 1)]
    result = {'schema_version': 2, 'taxonomy_version': 'test-only',
              'levels': ['primary', 'secondary', 'tertiary'], 'matching_mode': 'all_paths',
              'primary': [{'id': 'P1', 'name': 'Example group A', 'quota': dict(unlimited),
                           'secondaries': [secondary('S1'), secondary('S2', 257)]},
                          {'id': 'P2', 'name': 'Example group B', 'quota': dict(unlimited),
                           'secondaries': [secondary('S3', 2, True)]}],
              'rules': [], 'restrictions': {}, 'unknown_id': 'unclassified',
              'production_forms': forms,
              'production_form_priority': ['F05', 'F04', 'F01', 'F02', 'F03', 'F06', 'F07'],
              'embedding': {'model_name': 'Qwen3-Embedding-0.6B', 'dimensions': 1024,
                            'pooling': 'last_token_left_padding', 'max_length': 96,
                            'normalization': 'NFC_whitespace_2048chars_L2'},
              'quota_basis': 'effective_duration_seconds'}
    result['fingerprint'] = digest(result)
    return result
