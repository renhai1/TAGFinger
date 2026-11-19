"""Verification metrics of TAGFinger (Sec. 5.1 and Appendix B).

VSR is evaluated under FPR constraints using the fixed, tri-valued decision
rule from the main text. No additional continuous score or threshold sweep
on the evaluation set is defined by this implementation.
"""

from utils.protocol import fixed_operating_point


def vsr_at_fpr(positive_decisions, negative_decisions, fpr_budget):
    """VSR at a fixed rule; None means its measured FPR exceeds the budget."""
    return fixed_operating_point(positive_decisions, negative_decisions,
                                  (fpr_budget,))["constrained_vsr"][0]


def s_at_k(positive_decisions, negative_decisions, budgets=(0.01, 0.05)):
    """Constrained fixed-rule VSR, not an interpolated ROC operating point."""
    return tuple(vsr_at_fpr(positive_decisions, negative_decisions, b)
                 for b in budgets)


def reconstruction_analysis(clean_x, perturbed_x, fingerprint_nodes,
                            device, epochs=50, num_normal=200):
    """Reconstruction loss distributions of generated perturbations vs.
    normal data (Figure 6(c,d)): an autoencoder fitted on clean attributes
    should reconstruct perturbed fingerprint attributes as well as normal
    ones, indicating that the perturbations are indistinguishable from
    normal data and do not compromise dataset quality.
    """
    import torch
    from model.reconstruct import MLPAE
    num_normal = min(num_normal, clean_x.size(0))
    normal_idx = torch.randperm(clean_x.size(0))[:num_normal]
    ae = MLPAE(clean_x, perturbed_x[fingerprint_nodes], device, epochs)
    ae.fit()
    normal_errors = ae.inference(clean_x[normal_idx])
    perturbed_errors = ae.inference(perturbed_x[fingerprint_nodes])
    return normal_errors.cpu(), perturbed_errors.cpu()
