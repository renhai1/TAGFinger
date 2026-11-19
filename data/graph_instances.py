"""Original graph-classification instances stored as a disjoint union.

``graph_instances.pt`` contains graph_index (one original graph ID per node),
labels (one original class label per graph), num_classes, and graph-level
train_mask / val_mask / test_mask. These are source annotations, not labels
inferred from node classes, connected components, or fingerprint regions.
"""

import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph


GRAPH_FIELDS = ("graph_index", "graph_y", "graph_num_classes",
                "graph_train_mask", "graph_val_mask", "graph_test_mask")


def require_original_graphs(source):
    missing = [key for key in GRAPH_FIELDS if getattr(source, key, None) is None]
    if missing:
        raise ValueError(
            "The graph task requires original graph instances and labels in "
            "graph_instances.pt; node labels must not be converted to graph "
            "labels. Supply graph metadata or use --tasks node [edge].")


def attach_original_graphs(source, metadata):
    """Validate source annotations before attaching them to the node union."""
    if not isinstance(metadata, dict):
        raise ValueError("graph_instances.pt must contain a tensor dictionary")
    index = metadata.get("graph_index")
    labels = metadata.get("labels")
    if not isinstance(index, torch.Tensor) or index.dtype != torch.long:
        raise ValueError("graph_index must be a torch.long vector")
    if index.ndim != 1 or index.numel() != source.num_nodes:
        raise ValueError("graph_index must provide one original graph ID per node")
    if (not isinstance(labels, torch.Tensor) or labels.dtype != torch.long
            or labels.ndim != 1 or labels.numel() == 0):
        raise ValueError("labels must be a nonempty torch.long graph-label vector")
    count = labels.numel()
    if not torch.equal(torch.unique(index.cpu()), torch.arange(count)):
        raise ValueError("Original graph IDs must be contiguous and nonempty")
    classes = metadata.get("num_classes")
    if not isinstance(classes, int) or isinstance(classes, bool) or classes < 2:
        raise ValueError("num_classes must declare at least two graph classes")
    if bool((labels < 0).any()) or bool((labels >= classes).any()):
        raise ValueError("Original graph labels are outside their declared class space")
    masks = {}
    for split in ("train", "val", "test"):
        mask = metadata.get(split + "_mask")
        if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
                or mask.ndim != 1 or mask.numel() != count):
            raise ValueError(f"{split}_mask must have one boolean per original graph")
        masks[split] = mask.to(source.x.device)
    if not bool(masks["train"].any()):
        raise ValueError("At least one original graph must be in the training split")
    memberships = sum(mask.long() for mask in masks.values())
    if bool((memberships != 1).any()):
        raise ValueError("Original graph splits must be disjoint and cover all graphs")
    index = index.to(source.x.device)
    if source.edge_index.numel():
        src, dst = source.edge_index
        if bool((index[src] != index[dst]).any()):
            raise ValueError("Original graph instances cannot have cross-graph edges")
    source.graph_index = index
    source.graph_y = labels.to(source.x.device)
    source.graph_num_classes = classes
    for split, mask in masks.items():
        setattr(source, "graph_" + split + "_mask", mask)
    return source


def original_graph_nodes(source, graph_id):
    require_original_graphs(source)
    graph_id = int(graph_id)
    if not 0 <= graph_id < source.graph_y.numel():
        raise ValueError(f"Unknown original graph ID: {graph_id}")
    nodes = (source.graph_index == graph_id).nonzero(as_tuple=False).view(-1)
    if nodes.numel() == 0:
        raise ValueError(f"Original graph {graph_id} has no remaining nodes")
    return nodes


def original_graph(source, graph_id, x=None):
    """Return the entire original graph with its unchanged graph label.

    ``x`` may be the clean or PASP-edited feature union. Its node order must
    match ``source``. Selecting a fingerprint does not shrink the graph.
    """
    nodes = original_graph_nodes(source, graph_id)
    features = source.x if x is None else x
    if features.shape != source.x.shape:
        raise ValueError("Graph query features must match the original node union")
    edges, _ = subgraph(nodes, source.edge_index, relabel_nodes=True,
                        num_nodes=source.num_nodes)
    return Data(x=features[nodes], edge_index=edges,
                y=source.graph_y[int(graph_id)].view(1))


def training_graph_ids(source):
    require_original_graphs(source)
    # Node-removal attacks may delete a whole graph. Do not fabricate an input
    # for it, but preserve labels and splits for every surviving instance.
    present = torch.bincount(source.graph_index,
                             minlength=source.graph_y.numel()) > 0
    ids = (source.graph_train_mask & present).nonzero(as_tuple=False).view(-1)
    if ids.numel() == 0:
        raise ValueError("No original training graphs remain")
    return ids


def sample_original_graphs(source, count):
    ids = training_graph_ids(source)
    sampled = ids[torch.randint(ids.numel(), (count,), device=ids.device)]
    return [original_graph(source, int(graph_id)) for graph_id in sampled]


def copy_graph_annotations(source, target, node_ids=None):
    """Preserve graph identities/labels when an attack modifies node inputs."""
    if getattr(source, "graph_index", None) is None:
        return target
    require_original_graphs(source)
    for field in GRAPH_FIELDS:
        value = getattr(source, field)
        if field == "graph_index" and node_ids is not None:
            value = value[node_ids]
        setattr(target, field, value.clone() if isinstance(value, torch.Tensor)
                else value)
    return target
