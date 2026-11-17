"""Verification metrics of TAGFinger (Sec. 5.1 and Appendix B).

The verification success rate (VSR) is reported under false positive rate
(FPR) constraints: S@K denotes the mean VSR at FPR <= K%. Following
Proposition 5, the owner declares "violated" when the fraction of positive
evidence over the m fingerprint nodes is at least a threshold vartheta';
sweeping vartheta' traces the VSR/FPR curve, and S@K is the largest VSR whose
FPR does not exceed the budget.
"""

import numpy as np
import torch

from model.reconstruct import MLPAE


def vsr_at_fpr(pos_frac_suspected, pos_frac_independent, fpr_budget):
    """Largest VSR under FPR <= fpr_budget, sweeping vartheta'."""
    suspected = np.asarray(pos_frac_suspected, dtype=np.float64)
    independent = np.asarray(pos_frac_independent, dtype=np.float64)
    thresholds = np.unique(np.concatenate(
        [suspected, independent, [0.0]]))
    best_vsr = 0.0
    for t in thresholds:
        fpr = float((independent >= t).mean())
        if fpr <= fpr_budget:
            vsr = float((suspected >= t).mean())
            best_vsr = max(best_vsr, vsr)
    return best_vsr


def s_at_k(pos_frac_suspected, pos_frac_independent, budgets=(0.01, 0.05)):
    """S@1 and S@5 (VSR at FPR <= 1% and <= 5%)."""
    return tuple(vsr_at_fpr(pos_frac_suspected, pos_frac_independent, b)
                 for b in budgets)


def reconstruction_analysis(clean_x, perturbed_x, fingerprint_nodes,
                            device, epochs=50, num_normal=200):
    """Reconstruction loss distributions of generated perturbations vs.
    normal data (Figure 6(c,d)): an autoencoder fitted on clean attributes
    should reconstruct perturbed fingerprint attributes as well as normal
    ones, indicating that the perturbations are indistinguishable from
    normal data and do not compromise dataset quality.
    """
    num_normal = min(num_normal, clean_x.size(0))
    normal_idx = torch.randperm(clean_x.size(0))[:num_normal]
    ae = MLPAE(clean_x, perturbed_x[fingerprint_nodes], device, epochs)
    ae.fit()
    normal_errors = ae.inference(clean_x[normal_idx])
    perturbed_errors = ae.inference(perturbed_x[fingerprint_nodes])
    return normal_errors.cpu(), perturbed_errors.cpu()