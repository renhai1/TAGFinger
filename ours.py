"""TAGFinger ownership verification with explicit, paper-aligned protocols.

Importing this module or asking for --help does not train a model or load
optional tensor/LLM dependencies. Each audited model receives its own SAKA,
PASP, and prompt optimization. No surrogate/LLM gradients enter a black box.
Existing paper tables have not been revalidated by these implementation fixes.
"""
import argparse
from datetime import datetime
import importlib
import json
import logging
import math
from pathlib import Path
import random

DATASETS = ('Cora-ML', 'CiteSeer-ML', 'ArXiv-HepPh', 'ArXiv-CS', 'Reddit')
BACKBONES = ('GCN', 'GAT', 'GIN', 'RGCN')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model', choices=BACKBONES, default='GCN')
    parser.add_argument('--surrogate_model', choices=BACKBONES, default='GCN')
    parser.add_argument('--dataset', choices=DATASETS, default='Cora-ML')
    parser.add_argument('--mix_dataset', choices=DATASETS)
    parser.add_argument('--mix_ratio', type=float, default=.8)
    parser.add_argument('--data_root', default='data/TAG')
    parser.add_argument('--tasks', nargs='+', choices=('node', 'edge', 'graph'),
                        default=['node', 'edge', 'graph'],
                        help='graph requires genuine graph_instances.pt annotations; node is required')
    parser.add_argument('--theta', type=int,
                        help='Task vote threshold; default strict majority (three tasks: 2)')
    integer_options = {'hidden': 128, 'num_layer': 2, 'epochs': 200,
        'saka_epochs': 500, 'patience': 50, 'gen_hidden': 256,
        'batch_node': 128, 'batch_graph': 32, 'total_select': 200,
        'topk_stable': 2, 'pasp_rounds': 5, 'num_candidates': 4,
        'prompt_steps': 10, 'prompt_tokens': 10, 'num_hops': 2}
    float_options = {'train_lr': .01, 'weight_decay': 5e-4, 'dropout': .5,
        'saka_lr': 1e-3, 'x_budget': .05, 'edge_budget': .01,
        'delta_budget': .1, 'lam': .5, 'pasp_lr': 1e-3,
        'cross_prune': .1, 'inner_prune': .01, 'attack_ratio': .2}
    for name, default in integer_options.items():
        parser.add_argument('--' + name, type=int, default=default)
    for name, default in float_options.items():
        parser.add_argument('--' + name, type=float, default=default)
    parser.add_argument('--llm_path', help='Base or PEFT-adapter model directory')
    parser.add_argument('--text_encoder', help='The same encoder used to prepare clean features')
    parser.add_argument('--pert_cache', help='JSON node-ID to candidate-edit dictionary/list')
    parser.add_argument('--llm_mode', choices=('single', 'multi'))
    for name in ('no_dac', 'no_dvc', 'no_tup', 'recon_analysis'):
        parser.add_argument('--' + name, action='store_true')
    parser.add_argument('--attack', choices=('none', 'prune', 'od', 'rigbd', 'mad'), default='none')
    parser.add_argument('--independent_data_root',
                        help='Optional prepared no-use training data; never synthesized from protected data')
    parser.add_argument('--independent_dataset', choices=DATASETS,
                        help='Folder name for the supplied no-use data; defaults to --dataset')
    parser.add_argument('--label_space_id',
                        help='Shared task label semantics, matching control provenance.json')
    parser.add_argument('--report_path', help='Output JSON report (default under logs/)')
    return parser


def validate_args(parser, args):
    args.tasks = tuple(tau for tau in ('node', 'edge', 'graph') if tau in args.tasks)
    if 'node' not in args.tasks:
        parser.error('node is required for surrogate regional predictions')
    if args.theta is None:
        args.theta = len(args.tasks) // 2 + 1
    if not 1 <= args.theta <= len(args.tasks):
        parser.error('theta must be between 1 and the selected task count')
    if args.no_dac and args.no_dvc:
        parser.error('SAKA needs at least one of DAC or DVC')
    for name in ('hidden', 'num_layer', 'epochs', 'saka_epochs', 'patience',
                 'gen_hidden', 'batch_node', 'batch_graph', 'total_select',
                 'pasp_rounds', 'num_candidates', 'prompt_steps', 'prompt_tokens'):
        if getattr(args, name) <= 0:
            parser.error(f'{name} must be positive')
    if args.num_layer < 2 or args.topk_stable < 0 or args.num_hops < 0:
        parser.error('num_layer >= 2, topk_stable >= 0, and num_hops >= 0 are required')
    for name in ('train_lr', 'saka_lr', 'delta_budget', 'pasp_lr'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f'{name} must be finite and positive')
    for name in ('weight_decay', 'x_budget', 'lam'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f'{name} must be finite and nonnegative')
    for name in ('dropout', 'edge_budget', 'cross_prune', 'inner_prune', 'attack_ratio'):
        if not math.isfinite(getattr(args, name)) or not 0 <= getattr(args, name) <= 1:
            parser.error(f'{name} must lie in [0, 1]')
    if not 0 < args.mix_ratio < 1:
        parser.error('mix_ratio must be strictly between zero and one')
    if not args.text_encoder or not (args.llm_path or args.pert_cache):
        parser.error('PASP requires --text_encoder and either --llm_path or --pert_cache')
    if args.independent_data_root and not args.label_space_id:
        parser.error('Supplied no-use data requires --label_space_id and provenance.json')
    return args


def seed_run(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_gnn(name, source, args, device):
    # Avoid eagerly importing RGCN/torch_sparse for a GCN-only experiment.
    cls = importlib.import_module('model').__getattr__(name)
    return cls(input_dim=source.x.size(1), hid_dim=args.hidden,
               num_layer=args.num_layer, drop_ratio=args.dropout).to(device)


def build_heads(source, args, device):
    from model.NodeClassifier import NodeClassifier
    from model.GraphHead import GraphClassifier
    classifier = NodeClassifier(hid_dim=args.hidden, num_classes=source.num_classes,
                                 dropout=args.dropout, inner_dim=args.hidden).to(device)
    classes = {'node': source.num_classes, 'edge': 2}
    if 'graph' in args.tasks:
        classes['graph'] = source.graph_num_classes
    heads = {tau: GraphClassifier(hid_dim=args.hidden, num_classes=classes[tau],
                                 inner_dim=args.hidden, dropout=args.dropout).to(device)
             for tau in args.tasks}
    # Node and node-centered graph predictions use the same class semantics.
    # Share their decoder so graph-posterior alignment also trains the local
    # surrogate probabilities used for region selection; no suspected node
    # outputs are needed by the task-unified audit.
    heads['node'].lin1, heads['node'].lin2 = classifier.lin1, classifier.lin2
    return classifier, heads


def sample_graph_batch(tau, source, num_instances, args, device):
    import torch
    from torch_geometric.data import Batch
    from torch_geometric.utils import negative_sampling
    from data.graph_instances import sample_original_graphs
    from prompt.TUP import reformulate
    if tau == 'graph':
        batch = Batch.from_data_list(sample_original_graphs(source, num_instances)).to(device)
        return batch, batch.y.view(-1)
    train_mask = source.train_mask.clone()
    if 'graph' in args.tasks:
        train_mask &= source.graph_train_mask[source.graph_index]
    train_idx = train_mask.nonzero(as_tuple=False).view(-1)
    if not train_idx.numel():
        raise ValueError('No task-training centers remain')
    train_edges = (train_mask[source.edge_index[0]] & train_mask[source.edge_index[1]]
                   & (source.edge_index[0] != source.edge_index[1]))
    edge_ids = train_edges.nonzero(as_tuple=False).view(-1)
    graphs, labels = [], []
    for i in range(num_instances):
        if tau == 'node':
            node = int(train_idx[torch.randint(train_idx.numel(), (1,), device=device)].item())
            graph = reformulate('node', node, source.x, source.edge_index, args.num_hops)
            label = int(source.y[node])
        elif tau == 'edge':
            if i % 2 == 0:
                if not edge_ids.numel():
                    raise ValueError('No positive training edges with training endpoints')
                edge = int(edge_ids[torch.randint(edge_ids.numel(), (1,), device=device)].item())
                u, v = source.edge_index[:, edge].tolist()
                label = 1
            else:
                negatives = negative_sampling(source.edge_index,
                    num_nodes=source.num_nodes, num_neg_samples=max(64, num_instances))
                allowed = train_mask[negatives[0]] & train_mask[negatives[1]]
                if getattr(source, 'graph_index', None) is not None:
                    allowed &= source.graph_index[negatives[0]] == source.graph_index[negatives[1]]
                negatives = negatives[:, allowed]
                if not negatives.size(1):
                    raise ValueError('No valid negative training edges; use a sparser graph/split')
                u, v = negatives[:, 0].tolist()
                label = 0
            graph = reformulate('edge', (u, v), source.x, source.edge_index, args.num_hops)
        else:
            raise ValueError(f'Unknown task: {tau}')
        graphs.append(graph)
        labels.append(label)
    return Batch.from_data_list(graphs).to(device), torch.tensor(labels, device=device)


def train_blackbox(source, args, device, tag, log):
    import torch
    from model.SAKA import BlackBoxGNN
    gnn = build_gnn(args.model, source, args, device)
    classifier, heads = build_heads(source, args, device)
    parameters = list(gnn.parameters()) + list(classifier.parameters())
    for head in heads.values():
        parameters.extend(head.parameters())
    parameters = list({id(p): p for p in parameters}.values())
    optimizer = torch.optim.Adam(parameters, lr=args.train_lr, weight_decay=args.weight_decay)
    train_mask = source.train_mask.clone()
    if 'graph' in args.tasks:
        train_mask &= source.graph_train_mask[source.graph_index]
    indices = train_mask.nonzero(as_tuple=False).view(-1)
    if not indices.numel():
        raise ValueError('Training split is empty after attack/graph-split restrictions')
    for _ in range(args.epochs):
        for module in (gnn, classifier, *heads.values()):
            module.train()
        optimizer.zero_grad()
        logits, _ = classifier(gnn(source.x, source.edge_index))
        nodes = indices[torch.randint(indices.numel(), (args.batch_node,), device=device)]
        loss = torch.nn.functional.cross_entropy(logits[nodes], source.y[nodes])
        for tau in args.tasks:
            batch, labels = sample_graph_batch(tau, source, args.batch_graph, args, device)
            # GNN returns node embeddings; the task head owns graph readout.
            task_logits, _ = heads[tau](gnn(batch.x, batch.edge_index), batch=batch.batch)
            loss = loss + torch.nn.functional.cross_entropy(task_logits, labels)
        loss.backward()
        optimizer.step()
    for module in (gnn, classifier, *heads.values()):
        module.eval()
    log(f'[{tag}] training complete; task heads use their own label spaces')
    return BlackBoxGNN(gnn, classifier, heads)


def load_control(data, args, device, protected_texts):
    from data.tag_data import load_tag
    from data.graph_instances import require_original_graphs
    from data.control import validate_control_provenance
    name = args.independent_dataset or args.dataset
    folder = Path(args.independent_data_root) / name
    protected_folders = [Path(args.data_root) / args.dataset]
    if args.mix_dataset:
        protected_folders.append(Path(args.data_root) / args.mix_dataset)
    if any(folder.resolve() == p.resolve() for p in protected_folders):
        raise ValueError('No-use data must not reuse a protected data directory')
    with (folder / 'provenance.json').open() as stream:
        metadata = json.load(stream)
    control = load_tag(name, root=args.independent_data_root, seed=args.seed,
                       num_classes=data.num_classes).to(device)
    if control.x.size(1) != data.x.size(1):
        raise ValueError('Control inputs must use the same encoder/feature dimension')
    if 'graph' in args.tasks:
        require_original_graphs(control)
        if control.graph_num_classes != data.graph_num_classes:
            raise ValueError('Control graph labels must match the audited graph class space')
    provenance = validate_control_provenance(protected_texts, control.raw_texts,
        metadata, [args.dataset] + ([args.mix_dataset] if args.mix_dataset else []),
        args.label_space_id)
    return control, provenance


def main(argv=None):
    parser = build_parser()
    args = validate_args(parser, parser.parse_args(argv))
    import torch
    from data.tag_data import load_tag, mix_tag_datasets
    from data.graph_instances import require_original_graphs
    from utils.attacks import apply_attack
    from utils.audit import OwnershipAudit
    from utils.text_encoder import TextEncoder
    from utils.protocol import fixed_operating_point
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    Path('logs').mkdir(exist_ok=True)
    logging.basicConfig(filename=f'logs/{stamp}.log', level=logging.INFO)
    def log(message):
        print(message)
        logging.info(message)
    device = torch.device(f'cuda:{args.device_id}'
                          if not args.no_cuda and torch.cuda.is_available() else 'cpu')
    seed_run(args.seed)
    data = load_tag(args.dataset, root=args.data_root, seed=args.seed)
    protected_texts = list(data.raw_texts)
    if 'graph' in args.tasks:
        require_original_graphs(data)
    if args.mix_dataset:
        other = load_tag(args.mix_dataset, root=args.data_root, seed=args.seed)
        protected_texts.extend(other.raw_texts)
        data = mix_tag_datasets(data, other, args.mix_ratio, seed=args.seed,
                               preserve_graphs='graph' in args.tasks)
        if 'graph' in args.tasks:
            log('[Data] graph mixture ratio selects whole graph instances, not nodes')
    data = data.to(device)
    control, provenance = load_control(data, args, device, protected_texts) if args.independent_data_root else (None, None)
    encoder = TextEncoder(args.text_encoder, device=str(device))
    perturbator = None
    if args.llm_path:
        from llm.generator import LLMPerturbator
        perturbator = LLMPerturbator(args.llm_path, device=str(device))
    cache = None
    if args.pert_cache:
        with open(args.pert_cache) as stream:
            cache = json.load(stream)
        if not isinstance(cache, dict):
            raise ValueError('Candidate cache must be a node-ID dictionary')
    reports = {}
    for tag, training_data in [('suspected', apply_attack(data, args.attack, args.attack_ratio)),
                               ('no_use', control)]:
        if training_data is None:
            continue
        seed_run(args.seed)
        blackbox = train_blackbox(training_data, args, device, tag, log)
        seed_run(args.seed)
        surrogate = build_gnn(args.surrogate_model, data, args, device)
        classifier, heads = build_heads(data, args, device)
        audit = OwnershipAudit(data, args, args.tasks, device, encoder,
            perturbator=perturbator, cached_candidates=cache, log_fn=log)
        audit.align(blackbox, surrogate, classifier, heads)
        audit.select_regions()
        audit.construct()
        reports[tag] = audit.verify(blackbox)
        reports[tag]['edits'] = {str(n): t for n, t in audit.edits.items()}
        if tag == 'suspected' and args.recon_analysis:
            from evaluation.metrics import reconstruction_analysis
            normal, edited = reconstruction_analysis(data.x, data.x + audit.delta,
                                                     audit.nodes, device)
            reports[tag]['reconstruction_means'] = [float(normal.mean()), float(edited.mean())]
        del audit, blackbox, surrogate, classifier, heads
    operating_points = {}
    if 'no_use' in reports:
        for mode in ('single', 'multi'):
            operating_points[mode] = fixed_operating_point(
                reports['suspected']['decisions'][mode], reports['no_use']['decisions'][mode])
        for tau in args.tasks:
            operating_points['task_' + tau] = fixed_operating_point(
                reports['suspected']['task_decisions'][tau], reports['no_use']['task_decisions'][tau])
    else:
        log('[Metrics] no supplied no-use data: FPR and S@K are unavailable')
    output = {'configuration': vars(args), 'audits': reports,
              'operating_points': operating_points, 'control_provenance': provenance,
              'protocol': 'model-specific SAKA/PASP; fixed tri-valued task votes',
              'paper_table_reproduction': False,
              'limitations': 'Fixed-rule S@1/S@5 may coincide; null means unmet FPR constraint. '
                             'No unreported threshold calibration or previous paper numbers are inferred.'}
    path = Path(args.report_path or f'logs/{stamp}-report.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    log(f'[Report] {path}')
    return output


if __name__ == '__main__':
    main()
