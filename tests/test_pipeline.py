"""Contract and end-to-end tests. Synthetic vectors; no model or video download."""
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from video_taxonomy.core import UNKNOWN, choose, choose_leaf, duration_seconds, eligibility, evidence, form_result, quota_status, video_key
from video_taxonomy.taxonomy import centers, digest, load, validate
from fixtures import sample_config
from video_taxonomy.pipeline import Matcher, TagLookup, run_config
from argparse import Namespace


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = sample_config()
        cls.config_dir = tempfile.TemporaryDirectory()
        cls.config_path = Path(cls.config_dir.name) / 'example.json'
        cls.config_path.write_text(json.dumps(cls.config))
        cls.secondary = {s['id']: s for p in cls.config['primary'] for s in p['secondaries']}

    @classmethod
    def tearDownClass(cls):
        cls.config_dir.cleanup()

    def test_external_configuration(self):
        self.assertEqual(load(self.config_path), self.config)
        self.assertEqual(len(centers(self.config)), 260)
        self.assertTrue(all(s['id'] not in {x['id'] for x in centers(self.config)}
                            for s in self.secondary.values() if s['children']))

    def test_only_separator_changes(self):
        a, b = centers(self.config), centers(self.config, ' ')
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            self.assertEqual(y['text'], ' '.join(x['path']))
            self.assertEqual({k: v for k, v in x.items() if k != 'text'}, {k: v for k, v in y.items() if k != 'text'})

    def test_saved_separator_and_cache_isolation(self):
        with tempfile.TemporaryDirectory() as temp:
            args = Namespace(config=self.config_path, run_dir=Path(temp), path_separator='space')
            _, first = run_config(args, create=True)
            self.assertEqual(first['path_separator'], ' ')
            args.path_separator = None
            self.assertEqual(run_config(args)[1], first)
            args.path_separator = 'slash'
            with self.assertRaises(ValueError):
                run_config(args)

    def test_corruption_and_quotas(self):
        broken = json.loads(json.dumps(self.config))
        broken['primary'][0]['name'] = 'changed'
        with self.assertRaises(ValueError):
            validate(broken)
        quota = {'min_share': .1, 'max_share': .9, 'min_hours': 10}
        self.assertEqual(quota_status(.2, 20, quota, False), 'not_evaluable')
        self.assertEqual(quota_status(.2, 9, quota, True), 'below_minimum')
        self.assertEqual(quota_status(.2, 20, quota, True), 'within_range')

    def test_vote_order_and_valid_parent(self):
        mapping = {'a': 'P1', 'b': 'P1', 'c': 'P2'}
        order = {'a': 0, 'b': 1, 'c': 2}
        self.assertEqual(choose([('a', .1), ('b', .2), ('c', .99)], mapping, order), ('P1', 'b'))
        self.assertEqual(choose([('a', .2), ('b', .2)], mapping, order), ('P1', 'a'))
        self.assertEqual(choose([], mapping, order), (UNKNOWN, UNKNOWN))
        self.assertEqual(choose([('a', float('nan'))], mapping, order), (UNKNOWN, UNKNOWN))

    def test_evidence_and_duration(self):
        self.assertEqual(evidence({'tags': ['example-a', 'example-a', '', '12345', '🔥'], 'title': 'example-a'}), ['example-a'])
        self.assertEqual(evidence({'tags': 'example-a,example-b'}), ['example-a', 'example-b'])
        self.assertEqual(duration_seconds({'duration_seconds': 10}), (10, 'raw'))
        self.assertEqual(duration_seconds({'effective_duration_seconds': 10}), (10, 'effective'))
        self.assertEqual(duration_seconds({'effective_duration_seconds': float('nan')}), (None, 'invalid'))
        self.assertEqual(duration_seconds({}), (None, 'missing'))

    def test_leaf_vote_preserves_parents(self):
        leaves = [
            {'id': 'a1', 'secondary_id': 'a', 'primary_id': 'P1'},
            {'id': 'a2', 'secondary_id': 'a', 'primary_id': 'P1'},
            {'id': 'b1', 'secondary_id': 'b', 'primary_id': 'P1'},
            {'id': 'c', 'secondary_id': 'c', 'primary_id': 'P2'},
        ]
        self.assertEqual(choose_leaf([('a1', .7), ('a2', .8), ('b1', .99), ('c', 1.)], leaves), ('P1', 'a', 'a2'))
        self.assertEqual(choose_leaf([('a2', .7), ('a1', .7)], leaves), ('P1', 'a', 'a1'))
        self.assertEqual(choose_leaf([], leaves), (UNKNOWN, UNKNOWN, UNKNOWN))

    def test_forms_are_separate_and_prioritized(self):
        forms = {f['name']: f['id'] for f in self.config['production_forms']}
        result = form_result({'production_forms': ['专业影视与节目', '动画与 CG', '游戏画面']}, self.config)
        self.assertEqual(result[0], forms['游戏画面'])
        self.assertEqual(result[1], 'provided')
        self.assertEqual(form_result({'tags': ['music', 'game', 'animation']}, self.config)[0], UNKNOWN)
        self.assertEqual(form_result({'title': '今天真开心'}, self.config)[0], UNKNOWN)
        self.assertEqual(form_result({'title': 'vlog'}, self.config)[0], forms['实拍记录'])
        with self.assertRaises(ValueError):
            form_result({'production_form': '慢动作'}, self.config)

    def test_identity_and_restrictions(self):
        self.assertNotEqual(video_key({'id': '1'}, 'youtube'), video_key({'id': '1'}, 'tiktok'))
        self.assertEqual(video_key({'url': 'https://youtu.be/abc'}, 'bvd'), video_key({'url': 'https://www.youtube.com/watch?v=abc'}, 'bvd'))
        self.assertIsNone(video_key({}, 'test'))
        self.assertEqual(eligibility({'is_aigc': True}, None), 'excluded_aigc')
        self.assertEqual(eligibility({'is_aigc': 'false'}, None), 'not_excluded_by_metadata')
        restricted = next(s for s in self.secondary.values() if s['restricted'])
        self.assertEqual(eligibility({'is_aigc': False}, restricted), 'restricted_pending_review')


class EndToEnd(unittest.TestCase):
    def test_new_centers_cached_tags_stream_and_aggregate(self):
        config = sample_config()
        spec = {'taxonomy_fingerprint': config['fingerprint'], 'embedding': config['embedding'], 'centers': centers(config)}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_path = root / 'example.json'
            config_path.write_text(json.dumps(config))
            run = root / 'run'
            run.mkdir()
            (run / 'run.json').write_text(json.dumps(spec))
            secondary_ids = [s['id'] for p in config['primary'] for s in p['secondaries']]
            leaf_ids = [leaf['id'] for leaf in spec['centers']]
            vectors = np.zeros((len(spec['centers']), 1024), np.float16)
            for i, row in enumerate(spec['centers']):
                vectors[i, i] = 1
            np.save(run / 'center_vectors.npy', vectors)
            (run / 'center_vectors.json').write_text(json.dumps({'run_fingerprint': digest(spec), 'sha256': hashlib.sha256((run / 'center_vectors.npy').read_bytes()).hexdigest()}))
            legacy = root / 'legacy'
            (legacy / 'vectors').mkdir(parents=True)
            (legacy / 'center_encoding.json').write_text(json.dumps({'dimensions': 1024, 'model': '/models/Qwen3-Embedding-0.6B'}))
            tags = ['example-a', 'example-b', '杂音']
            target = [secondary_ids[0], secondary_ids[-1]]
            source = root / 'tags.parquet'
            pq.write_table(pa.table({'tag': tags}), source, row_group_size=2)
            x = np.zeros((3, 1024), np.float16)
            x[0, leaf_ids.index(target[0])] = 1
            x[1, -1 + len(leaf_ids)] = 1
            np.save(legacy / 'vectors/000000.npy', x[:2])
            np.save(legacy / 'vectors/000001.npy', x[2:])

            def invoke(*args, ok=True):
                result = subprocess.run([sys.executable, '-m', 'video_taxonomy', '--config', str(config_path), *map(str, args)], capture_output=True, text=True)
                if ok:
                    self.assertEqual(result.returncode, 0, result.stderr)
                else:
                    self.assertNotEqual(result.returncode, 0)
                return result

            invoke('assign-tags', '--run-dir', run, '--legacy-path-vectors', legacy, '--tag-source', source)
            args = Namespace(run_dir=run, min_score=0., min_margin=0.)
            lookup = TagLookup(run / 'tag_lookup', args, config, spec)
            self.assertEqual(lookup.get(tags[0])[0], target[0])
            self.assertEqual(lookup.get(tags[1])[0], leaf_ids[-1])  # >255 label code
            self.assertEqual(lookup.get(tags[2])[0], UNKNOWN)  # inherited zero/noise vector
            self.assertIsNone(lookup.get('未收录'))
            # Full 128-bit collision lookup: same first hash, different second.
            a = lookup.arrays
            text_hash = np.frombuffer(hashlib.md5(tags[0].encode()).digest(), dtype='<u8')
            lookup.arrays = {'h1': np.array([text_hash[0], text_hash[0]], dtype='<u8'),
                             'h2': np.array([text_hash[1] ^ np.uint64(1), text_hash[1]], dtype='<u8'),
                             'labels': np.array([0, 1]), 'scores': np.array([0., 1.]), 'margins': np.array([0., 1.])}
            self.assertEqual(lookup.get(tags[0])[0], leaf_ids[0])
            lookup.arrays = a
            rows = [
                {'id': 'a', 'tags': [tags[0], tags[0]], 'effective_duration_seconds': 10, 'is_aigc': False},
                {'id': 'a', 'tags': [tags[1]], 'effective_duration_seconds': 50},
                {'id': 'b', 'tags': [tags[1]], 'duration_seconds': 20, 'production_forms': ['动画与 CG', '游戏画面'], 'is_aigc': True},
                {'id': 'c', 'title': '12345', 'tags': [tags[2]]},
                {'id': 'd', 'tags': ['未收录']},
                {'id': 'e', 'tags': [tags[0]], 'content_secondary_id': target[0], 'production_form': '实拍记录', 'is_aigc': False, 'effective_duration_seconds': 3600},
            ]
            source = root / 'videos.jsonl'
            source.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
            output = root / 'classified.jsonl'
            invoke('classify', '--run-dir', run, '--input', source, '--platform', 'test', '--tag-lookup', run / 'tag_lookup', '--lookup-only', '--output', output)
            result = [json.loads(l) for l in output.read_text().splitlines()]
            self.assertEqual(len(result), 5)
            self.assertEqual(result[0]['secondary_id'], target[0])
            self.assertEqual(result[0]['evidence_count'], 1)
            self.assertEqual(result[0]['leaf_id'], target[0])
            self.assertEqual(result[0]['leaf_level'], 2)
            self.assertIsNone(result[0]['tertiary_id'])
            self.assertEqual(result[1]['leaf_id'], leaf_ids[-1])
            self.assertEqual(result[1]['leaf_level'], 3)
            self.assertEqual(result[1]['tertiary_id'], leaf_ids[-1])
            self.assertEqual(result[1]['secondary_id'], secondary_ids[-1])
            self.assertEqual(result[1]['production_form_id'], 'F05')
            self.assertEqual(result[1]['eligibility'], 'excluded_aigc')
            self.assertEqual(result[2]['secondary_id'], UNKNOWN)
            self.assertEqual(result[3]['secondary_id'], UNKNOWN)
            summary = root / 'summary.json'
            invoke('aggregate', '--input', output, '--output', summary)
            report = json.loads(summary.read_text())
            self.assertEqual(report['included_videos']['test'], 5)
            for dim in ['primary', 'secondary', 'leaf', 'production_form']:
                self.assertEqual(sum(r['video_count'] for r in report['rows'] if r['dimension'] == dim), 5)
            self.assertTrue(all(r['quota_status'] in {'not_evaluable', 'not_applicable'} for r in report['rows']))
            accepted = root / 'accepted.json'
            invoke('aggregate', '--input', output, '--accepted-only', '--output', accepted)
            report = json.loads(accepted.read_text())
            self.assertEqual(report['included_videos']['test'], 1)
            self.assertEqual(report['effective_duration_totals']['test'], 3600)
            invoke('aggregate', '--input', output, output, '--output', root / 'bad.json', ok=False)
            mixed = root / 'other-separator.jsonl'
            mixed.write_text(output.read_text())
            meta = json.loads(output.with_suffix('.jsonl.meta.json').read_text())
            meta['run_fingerprint'] = 'different separator'
            mixed.with_suffix('.jsonl.meta.json').write_text(json.dumps(meta))
            error = invoke('aggregate', '--input', output, mixed, '--output', root / 'mixed.json', ok=False)
            self.assertIn('Different center representations', error.stderr)
            invoke('classify', '--run-dir', run, '--input', source, '--platform', 'test', '--tag-lookup', run / 'tag_lookup', '--lookup-only', '--output', output, ok=False)
            # Changed centers are rejected instead of accidentally using old caches.
            with (run / 'center_vectors.npy').open('ab') as f:
                f.write(b'changed')
            with self.assertRaises(ValueError):
                Matcher(args, config, spec)


if __name__ == '__main__':
    unittest.main()
