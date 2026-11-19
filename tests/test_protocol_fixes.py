"""Protocol regression tests; dependency-free tests are never mocked metrics.

PyTorch tests are separately marked when tensor dependencies are unavailable.
"""
import ast
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.protocol import fingerprint_evidence, ownership_decision, fixed_operating_point
from evaluation.metrics import s_at_k
from data.control import validate_control_provenance
from llm.output import parse_edits
from ours import build_parser, validate_args

HAS_TENSORS = (importlib.util.find_spec('torch') is not None
               and importlib.util.find_spec('torch_geometric') is not None)


class ProtocolTests(unittest.TestCase):
    def test_evidence_cases(self):
        self.assertEqual(fingerprint_evidence(1, 0, 1), 1)
        self.assertEqual(fingerprint_evidence(1, 1, 1), 0)
        self.assertEqual(fingerprint_evidence(0, 0, 1), 0)
        self.assertEqual(fingerprint_evidence(2, 0, 1), -1)
        self.assertEqual(fingerprint_evidence(0, 1, 1), -1)

    def test_majority_and_uncertainty(self):
        self.assertEqual(ownership_decision([1, 1, 0], 2), 'violated')
        self.assertEqual(ownership_decision([-1, -1, 0], 2), 'unviolated')
        self.assertEqual(ownership_decision([1, 0, 0], 2), 'uncertain')
        self.assertEqual(ownership_decision([1, -1, 0], 1), 'violated')

    def test_unmet_fpr_is_not_zero_or_fake_threshold(self):
        positive = ['violated'] * 95 + ['uncertain'] * 5
        negative = ['violated'] * 2 + ['uncertain'] * 98
        self.assertEqual(s_at_k(positive, negative), (None, .95))
        report = fixed_operating_point(positive, negative)
        self.assertEqual(report['vsr'], .95)
        self.assertEqual(report['fpr'], .02)
        self.assertIn('null', json.dumps(report))

    def test_both_budgets_share_the_same_fixed_rule(self):
        positive = ['violated'] * 1908 + ['unviolated'] * 92
        negative = ['violated'] * 10 + ['uncertain'] * 1990
        self.assertEqual(s_at_k(positive, negative), (.954, .954))

    def test_invalid_metric_inputs_are_rejected(self):
        for positive, negative in (([], ['uncertain']), ([0., 1.], [0.]),
                                    (['violated'], ['unknown'])):
            with self.assertRaises(ValueError):
                fixed_operating_point(positive, negative)

    def test_control_needs_disjoint_sources_and_matching_semantics(self):
        valid = {'training_source_ids': ['external-A'], 'label_space_id': 'topic-v1'}
        result = validate_control_provenance(['protected text'], ['other text'],
                                              valid, ['Cora-ML'], 'topic-v1')
        self.assertEqual(result['training_source_ids'], ['external-A'])
        for metadata in ({}, dict(valid, training_source_ids=['Cora-ML']),
                         dict(valid, label_space_id='different')):
            with self.assertRaises(ValueError):
                validate_control_provenance(['a'], ['b'], metadata, ['Cora-ML'], 'topic-v1')

    def test_control_cannot_reuse_reformatted_protected_text(self):
        metadata = {'training_source_ids': ['other'], 'label_space_id': 'shared'}
        with self.assertRaisesRegex(ValueError, 'overlap'):
            validate_control_provenance(['secret  text'], ['secret\ntext'], metadata,
                                         ['protected'], 'shared')

    def test_strict_edit_parser_does_not_accept_explanations(self):
        self.assertEqual(parse_edits('Here is an explanation.', 4), {})
        self.assertEqual(parse_edits('<think>Node 4: wrong</think>\nNode 4: edited'),
                         {4: 'edited'})
        self.assertEqual(parse_edits('<think>Node 4: unfinished'), {})
        self.assertEqual(parse_edits('Node 4: one\nNode 4: two'), {})
        self.assertEqual(parse_edits('Node 4: first\nNode 5：second'), {4: 'first', 5: 'second'})

    def test_cli_help_does_not_load_tensor_or_llm_dependencies(self):
        result = subprocess.run([sys.executable, str(ROOT / 'ours.py'), '--help'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--independent_data_root', result.stdout)

    def test_cli_validation_and_default_threshold(self):
        parser = build_parser()
        args = parser.parse_args(['--tasks', 'node', '--text_encoder', 'encoder',
                                  '--pert_cache', 'edits.json'])
        self.assertEqual(validate_args(parser, args).theta, 1)
        for extras in (['--no_dac', '--no_dvc'], ['--theta', '4'], ['--delta_budget', '-1'],
                       ['--independent_data_root', 'external'], ['--mix_ratio', 'nan']):
            with self.subTest(extras=extras), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                args = parser.parse_args(['--text_encoder', 'encoder', '--pert_cache', 'edits.json'] + extras)
                validate_args(parser, args)

    def test_each_model_runs_its_own_complete_audit(self):
        source = (ROOT / 'ours.py').read_text()
        self.assertNotIn('make_independent_data', source)
        tree = ast.parse(source)
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        loops = [n for n in ast.walk(main) if isinstance(n, ast.For)]
        model_loop = next(n for n in loops if 'training_data' in ast.unparse(n.target))
        body = ast.unparse(model_loop)
        for operation in ('audit.align(', 'audit.select_regions()', 'audit.construct()', 'audit.verify('):
            self.assertIn(operation, body)
        self.assertNotIn('verify_ownership(independent', source)

    def test_no_feature_clipping_or_identity_candidate_fallback(self):
        source = (ROOT / 'utils/audit.py').read_text()
        self.assertNotIn('budget / norm', source)
        self.assertIn('No valid nonzero text candidate', source)
        self.assertIn('does not reproduce the prepared clean features', source)

    def test_saka_clean_query_and_validation_modes_wired(self):
        source = (ROOT / 'model/SAKA.py').read_text()
        self.assertIn('p_sus_clean = blackbox.query(data.x, data.edge_index)', source)
        self.assertIn('train_mask, val_mask = data.train_mask.clone(), data.val_mask.clone()', source)
        self.assertIn('val_pool = val_mask.nonzero', source)
        self.assertIn('node_pool=train_pool, graph_pool=train_graphs', source)
        validation = source.split('# early stopping on the validation alignment loss', 1)[1]
        self.assertIn('for head in graph_heads.values():\n            head.eval()', validation)
        self.assertIn('remove_query_edge(sub_ei_t, subset, query_edge)', source)

    def test_dvc_default_path_executes_without_undefined_clean_output(self):
        # Execute the real helper's control flow, with arithmetic doubles.
        # Tensor numerical correctness is tested separately below.
        tree = ast.parse((ROOT / 'model/SAKA.py').read_text())
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                    and n.name == 'alignment_loss')
        class Tensor:
            def __getitem__(self, _):
                return self
            def sum(self):
                return self
            def detach(self):
                return self
            def __mul__(self, _):
                return self
            __rmul__ = __mul__
            def __add__(self, _):
                return self
        calls = []
        def query(x, edges):
            calls.append(x)
            return Tensor()
        namespace = {
            'blackbox': SimpleNamespace(query=query),
            'sur_node_probs': lambda *_: Tensor(),
            'torch': SimpleNamespace(no_grad=contextlib.nullcontext),
            'data': SimpleNamespace(x=Tensor(), edge_index=Tensor()),
            'use_dac': False, 'use_dvc': True, 'graph_heads': {},
            'dvc_loss': lambda *_: Tensor(),
        }
        module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
        exec(compile(module, 'alignment_loss_test', 'exec'), namespace)
        namespace['alignment_loss'](Tensor(), Tensor(), [0], [], False)
        self.assertEqual(len(calls), 2)
        self.assertIs(calls[0], namespace['data'].x)

    def test_task_unified_audit_does_not_require_node_posteriors(self):
        tree = ast.parse((ROOT / 'model/SAKA.py').read_text())
        loss = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                    and n.name == 'alignment_loss')
        legacy = next(n for n in loss.body if isinstance(n, ast.If)
                      and ast.unparse(n.test) == 'not graph_heads')
        node_calls = [n for n in ast.walk(loss) if isinstance(n, ast.Call)
                      and ast.unparse(n.func) == 'blackbox.query']
        self.assertTrue(node_calls)
        self.assertTrue(all(n in list(ast.walk(legacy)) for n in node_calls))
        build = (ROOT / 'ours.py').read_text()
        self.assertIn("heads['node'].lin1, heads['node'].lin2 = classifier.lin1, classifier.lin2", build)


@unittest.skipUnless(HAS_TENSORS, 'requires torch and torch_geometric')
class TensorFixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global torch, Data
        import torch
        from torch_geometric.data import Data

    def data(self):
        return Data(x=torch.tensor([[.2, .8], [.3, .7], [.8, .2], [.7, .3]]),
                    edge_index=torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]),
                    y=torch.tensor([0, 0, 1, 1]),
                    train_mask=torch.tensor([True, True, False, False]),
                    val_mask=torch.tensor([False, False, True, True]),
                    test_mask=torch.zeros(4, dtype=torch.bool))

    def test_zero_and_nonzero_structure_budgets(self):
        from model.SAKA import GAGGenerator
        data = self.data()
        for budget in (0., .1, 1.):
            generator = GAGGenerator(2, 4, .05, budget)
            for param in generator.edge_mlp.parameters():
                torch.nn.init.constant_(param, 1)
            for _ in range(5):
                x, edges = generator(data.x, data.edge_index)
                self.assertLessEqual(edges.size(1) - data.edge_index.size(1), int(budget * data.edge_index.size(1)))
                self.assertLessEqual(float((x - data.x).abs().max()), .050001)
                added = edges[:, data.edge_index.size(1):]
                self.assertFalse(bool((added[0] == added[1]).any()))
                self.assertEqual(added.size(1), torch.unique(added, dim=1).size(1))
                if budget == 0:
                    self.assertTrue(torch.equal(edges, data.edge_index))

    def test_generator_respects_original_graph_boundaries(self):
        from model.SAKA import GAGGenerator
        data = self.data()
        index = torch.tensor([0, 0, 1, 1])
        generator = GAGGenerator(2, 4, .05, 1.)
        _, edges = generator(data.x, data.edge_index, index)
        self.assertTrue(bool((index[edges[0]] == index[edges[1]]).all()))

    def test_alignment_removes_target_edge_and_respects_center_pool(self):
        from model.SAKA import sample_alignment_instances
        data = self.data()
        pool = torch.tensor([2, 3])
        for tau, nodes, edges, query in sample_alignment_instances(
                data, 1, 8, tasks=('edge',), node_pool=pool):
            self.assertEqual(tau, 'edge')
            self.assertEqual(set(query), {2, 3})
            pu = int((nodes == query[0]).nonzero()[0])
            pv = int((nodes == query[1]).nonzero()[0])
            self.assertFalse(bool((((edges[0] == pu) & (edges[1] == pv)) |
                                   ((edges[0] == pv) & (edges[1] == pu))).any()))

    def test_dvc_training_and_validation_are_runnable(self):
        from model.SAKA import train_saka, GAGGenerator
        from model.GraphHead import GraphClassifier
        data = self.data()
        net = torch.nn.Linear(2, 2)
        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(2, 2)
            def forward(self, x, edge_index):
                return self.layer(x)
        class Classifier(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(2, 2)
            def forward(self, x):
                return self.layer(x), x
        class BlackBox:
            def query(self, x, edge_index):
                raise AssertionError('Task-unified alignment must not require node posteriors')
            def query_graph(self, x, edge_index, tau):
                with torch.no_grad():
                    return torch.softmax(net(x), 1).mean(0)
        head = GraphClassifier(2, 2, dropout=.5)
        modes = []
        head.register_forward_pre_hook(lambda module, inputs: modes.append((torch.is_grad_enabled(), module.training)))
        train_saka(Backbone(), Classifier(), GAGGenerator(2, 4, .05, 0),
                   BlackBox(), data, graph_heads={'node': head}, epochs=2,
                   batch_nodes=2, batch_graphs=2, log_fn=lambda _: None)
        self.assertIn((False, False), modes)
        self.assertNotIn((False, True), modes)

    def test_exact_text_features_and_over_budget_rejection(self):
        from utils.audit import text_candidate_delta
        data = self.data()
        data.raw_texts = ['a', 'b', 'c', 'd']
        class Encoder:
            def encode(self, texts):
                mapping = {text: data.x[i] for i, text in enumerate(data.raw_texts)}
                mapping['edited'] = data.x[0] + torch.tensor([.01, -.01])
                return torch.stack([mapping[text] for text in texts])
        delta = text_candidate_delta(data, Encoder(), {0: 'edited'}, .1, 'cpu')
        self.assertTrue(torch.allclose(data.x[0] + delta[0], Encoder().encode(['edited'])[0]))
        self.assertIsNone(text_candidate_delta(data, Encoder(), {0: 'edited'}, .001, 'cpu'))

    def test_protected_sampling_does_not_fill_budget_from_other_source(self):
        from utils.pasp import select_fingerprint_nodes
        data = self.data()
        scores = torch.tensor([.9, .8, float('-inf'), float('-inf')])
        allowed = torch.tensor([True, True, False, False])
        selected = select_fingerprint_nodes(scores, data.edge_index, 4, 20, 0, allowed)
        self.assertEqual(set(selected.tolist()), {0, 1})


if __name__ == '__main__':
    unittest.main()
