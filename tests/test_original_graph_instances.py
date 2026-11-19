"""Regression checks for original graph labels, training, alignment, queries.

No experiment is started by importing ours.py. Source wiring checks run with
the standard library; numerical graph tests require torch + torch_geometric.
"""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HAS_TENSORS = (importlib.util.find_spec("torch") is not None
               and importlib.util.find_spec("torch_geometric") is not None)


def function_node(path, name):
    tree = ast.parse((ROOT / path).read_text())
    return next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def load_function(path, name, namespace):
    node = function_node(path, name)
    # Test functions in isolation without importing the experiment runner.
    node.body = [statement for statement in node.body
                 if not isinstance(statement, (ast.Import, ast.ImportFrom))]
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, str(ROOT / path), "exec"), namespace)
    return namespace[name]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SourceWiringTests(unittest.TestCase):
    def test_graph_training_delegates_to_original_instances(self):
        node = function_node("ours.py", "sample_graph_batch")
        source = ast.unparse(node)
        self.assertIn("sample_original_graphs(source, num_instances)", source)
        self.assertIn("batch.y.view(-1)", source)
        self.assertNotIn("bincount", source)
        self.assertNotIn("num_hops + 1", source)

    def test_verification_uses_full_graph_not_stable_region(self):
        function = load_function("utils/audit.py", "make_task_instances", {
            "original_graph": lambda source, graph_id, x: (graph_id, x),
        })
        source = SimpleNamespace(graph_index=[0, 0, 0, 1, 1])
        clean, edited = object(), object()
        def query(x):
            return function(source, ("node", "edge", "graph"), 3, 4, x)
        self.assertEqual(query(clean)["graph"], (1, clean))
        self.assertEqual(query(edited)["graph"], (1, edited))
        self.assertEqual(query(clean)["node"], 3)
        self.assertEqual(query(clean)["edge"], (3, 4))

    def test_node_edge_mode_does_not_require_graph_annotations(self):
        function = load_function("utils/audit.py", "make_task_instances", {})
        self.assertEqual(function(object(), ("node", "edge"), 3, 4, object()),
                         {"node": 3, "edge": (3, 4)})

    def test_graph_class_count_is_not_node_class_count(self):
        source = (ROOT / "ours.py").read_text()
        self.assertIn("classes['graph'] = source.graph_num_classes", source)
        self.assertNotIn("'graph': source.num_classes", source)

    def test_alignment_does_not_crop_original_graphs(self):
        source = ast.unparse(function_node("model/SAKA.py",
                                           "sample_alignment_instances"))
        self.assertIn("original_graph_nodes", source)
        self.assertNotIn("num_hops + 1", source)
        self.assertIn("require_original_graphs", source)


@unittest.skipUnless(HAS_TENSORS, "requires torch and torch_geometric")
class TensorGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global torch, Data, Batch, subgraph, k_hop_subgraph
        import torch
        from torch_geometric.data import Data, Batch
        from torch_geometric.utils import subgraph, k_hop_subgraph
        cls.graphs = load_module("data/graph_instances.py", "graph_instance_tests")
        cls.tup = load_module("prompt/TUP.py", "tup_instance_tests")

    def source(self, graph_labels=(2, 0), classes=3):
        # Node-majority labels are 0 and 1, deliberately unlike graph labels.
        source = Data(x=torch.arange(10, dtype=torch.float).view(5, 2),
                      edge_index=torch.tensor([[0, 1, 1, 2, 3, 4],
                                               [1, 0, 2, 1, 4, 3]]),
                      y=torch.tensor([0, 0, 0, 1, 1]))
        source.train_mask = torch.ones(5, dtype=torch.bool)
        source.val_mask = torch.zeros(5, dtype=torch.bool)
        source.test_mask = torch.zeros(5, dtype=torch.bool)
        source.raw_texts = [str(i) for i in range(5)]
        source.num_classes = 2
        source.dataset_name = "fixture"
        metadata = {"graph_index": torch.tensor([0, 0, 0, 1, 1]),
                    "labels": torch.tensor(graph_labels), "num_classes": classes,
                    "train_mask": torch.tensor([True, False]),
                    "val_mask": torch.tensor([False, True]),
                    "test_mask": torch.tensor([False, False])}
        return source, metadata

    def attached(self):
        source, metadata = self.source()
        return self.graphs.attach_original_graphs(source, metadata)

    def test_graph_label_is_retained_not_node_majority(self):
        source = self.attached()
        graph = self.graphs.original_graph(source, 0)
        self.assertEqual(graph.y.item(), 2)
        self.assertEqual(graph.num_nodes, 3)
        self.assertTrue(torch.equal(graph.x, source.x[:3]))
        self.assertTrue(torch.equal(graph.edge_index, source.edge_index[:, :4]))

    def test_edit_changes_features_but_not_graph_shape_or_label(self):
        source = self.attached()
        edited = source.x.clone()
        edited[0] += 1
        edited.requires_grad_(True)
        clean = self.graphs.original_graph(source, 0)
        query = self.graphs.original_graph(source, 0, x=edited)
        self.assertEqual(query.num_nodes, clean.num_nodes)
        self.assertTrue(torch.equal(query.edge_index, clean.edge_index))
        self.assertTrue(torch.equal(query.y, clean.y))
        query.x.sum().backward()
        self.assertTrue(torch.equal(edited.grad[:3], torch.ones(3, 2)))
        self.assertEqual(edited.grad[3:].sum().item(), 0)

    def test_only_original_training_graphs_are_sampled(self):
        source = self.attached()
        graphs = self.graphs.sample_original_graphs(source, 10)
        self.assertEqual([g.y.item() for g in graphs], [2] * 10)
        self.assertEqual([g.num_nodes for g in graphs], [3] * 10)

    def test_training_batch_keeps_graph_label_space(self):
        source = self.attached()
        sample = load_function("ours.py", "sample_graph_batch", {
            "sample_original_graphs": self.graphs.sample_original_graphs,
            "Batch": Batch,
        })
        batch, labels = sample("graph", source, 2, SimpleNamespace(tasks=("graph",)),
                               torch.device("cpu"))
        self.assertEqual(labels.tolist(), [2, 2])
        self.assertEqual(batch.num_graphs, 2)
        self.assertEqual(batch.num_nodes, 6)

    def test_alignment_samples_entire_graph(self):
        source = self.attached()
        sample = load_function("model/SAKA.py", "sample_alignment_instances", {
            "torch": torch, "subgraph": subgraph,
            "k_hop_subgraph": k_hop_subgraph,
            "require_original_graphs": self.graphs.require_original_graphs,
            "original_graph_nodes": self.graphs.original_graph_nodes,
            "remove_query_edge": self.tup.remove_query_edge,
        })
        for tau, nodes, edges, _ in sample(source, 0, 12, tasks=("graph",)):
            graph_id = int(source.graph_index[nodes[0]])
            expected = self.graphs.original_graph(source, graph_id)
            self.assertEqual(tau, "graph")
            self.assertEqual(nodes.numel(), expected.num_nodes)
            self.assertTrue(torch.equal(edges, expected.edge_index))

    def test_phi_graph_is_identity_and_rejects_unlabeled_fallback(self):
        source = self.attached()
        original = self.graphs.original_graph(source, 0)
        self.assertIs(self.tup.reformulate("graph", original, source.x,
                                           source.edge_index), original)
        for bad in (0, Data(x=source.x, edge_index=source.edge_index)):
            with self.assertRaises(ValueError):
                self.tup.reformulate("graph", bad, source.x, source.edge_index)

    def test_missing_graph_metadata_fails_explicitly(self):
        source, _ = self.source()
        with self.assertRaisesRegex(ValueError, "graph_instances.pt"):
            self.graphs.sample_original_graphs(source, 1)

    def test_invalid_graph_annotations_fail(self):
        for field, value in (
            ("graph_index", torch.tensor([0, 0, 0, 2, 2])),
            ("graph_index", torch.tensor([0, 0, 1, 1, 1])),
            ("labels", torch.tensor([3, 0])),
            ("labels", torch.tensor([0, 0, 0, 1, 1])),
            ("train_mask", torch.tensor([True, True])),
            ("train_mask", torch.tensor([False, False])),
        ):
            with self.subTest(field=field, value=value):
                source, metadata = self.source()
                metadata[field] = value
                with self.assertRaises(ValueError):
                    self.graphs.attach_original_graphs(source, metadata)

    def test_node_removal_preserves_original_graph_label(self):
        source = self.attached()
        remove = load_function("utils/attacks.py", "_remove_nodes", {
            "subgraph": subgraph, "Data": Data,
            "copy_graph_annotations": self.graphs.copy_graph_annotations,
        })
        attacked = remove(source, torch.tensor([True, False, True, True, True]))
        graph = self.graphs.original_graph(attacked, 0)
        self.assertEqual(graph.y.item(), 2)
        self.assertEqual(graph.num_nodes, 2)

    def test_mixing_keeps_whole_graphs_and_separate_class_offsets(self):
        owner = self.attached()
        other = self.attached()
        # Both graph splits train here, so any whole-graph sample is usable.
        for source in (owner, other):
            source.graph_train_mask[:] = True
            source.graph_val_mask[:] = False
        mix = load_function("data/tag_data.py", "mix_tag_datasets", {
            "torch": torch, "subgraph": subgraph, "Data": Data,
            "require_original_graphs": self.graphs.require_original_graphs,
            "attach_original_graphs": self.graphs.attach_original_graphs,
        })
        result = mix(owner, other, 0.5, seed=4, preserve_graphs=True)
        self.assertEqual(result.graph_num_classes, 6)
        self.assertEqual(result.num_classes, 4)
        for graph_id in (0, 1):
            graph = self.graphs.original_graph(result, graph_id)
            source = owner if graph_id == 0 else other
            label = graph.y.item() - (0 if graph_id == 0 else 3)
            source_id = int((source.graph_y == label).nonzero()[0])
            expected = self.graphs.original_graph(source, source_id)
            self.assertTrue(torch.equal(graph.x, expected.x))
            self.assertTrue(torch.equal(graph.edge_index, expected.edge_index))


if __name__ == "__main__":
    unittest.main()
