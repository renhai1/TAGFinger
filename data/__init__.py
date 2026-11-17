from .induced_graph import induced_graphs, split_induced_graphs
from . graph_split import graph_split
from .Dataset import GraphDataset
from .tag_data import load_tag, mix_tag_datasets, TAG_DATASETS

# Legacy ProG-based loaders (baselines); optional for the TAGFinger pipeline.
try:
    from .load4data import load4graph, load4link_prediction_single_graph, load4node, load4link_prediction_multi_graph, NodePretrain, node_sample_and_save, graph_sample_and_save
except ImportError:
    pass
