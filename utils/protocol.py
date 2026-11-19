"""The main paper's tri-valued evidence and fixed task-voting rule.

No gradients, model internals, continuous ownership scores, or additional
cross-fingerprint threshold are introduced here.
"""

import math


def fingerprint_evidence(pred_pert, pred_clean, target):
    if pred_pert == target and pred_clean != target:
        return 1
    if pred_pert == pred_clean:
        return 0
    return -1


def ownership_decision(evidence, theta):
    evidence = list(evidence)
    if not evidence or not isinstance(theta, int) or not 1 <= theta <= len(evidence):
        raise ValueError("theta must be an integer between 1 and the task count")
    if any(value not in (-1, 0, 1) for value in evidence):
        raise ValueError("Evidence must belong to {-1, 0, 1}")
    # The main text orders the positive branch before the negative branch.
    # With non-majority theta both may be met; positive-first is intentional.
    if evidence.count(1) >= theta:
        return "violated"
    if evidence.count(-1) >= theta:
        return "unviolated"
    return "uncertain"


def decision_rates(decisions):
    decisions = list(decisions)
    if not decisions or any(d not in ("violated", "unviolated", "uncertain")
                            for d in decisions):
        raise ValueError("A nonempty collection of tri-valued decisions is required")
    return {key: decisions.count(key) / len(decisions)
            for key in ("violated", "unviolated", "uncertain")}


def fixed_operating_point(positive_decisions, negative_decisions, budgets=(.01, .05)):
    """Report VSR only if this fixed rule meets the measured FPR constraint.

    A failed constraint yields None (JSON null), not a fabricated zero or a
    different decision threshold chosen on the evaluation set. These are
    query-level empirical rates, not a claim about independent model trials.
    """
    positive = decision_rates(positive_decisions)
    negative = decision_rates(negative_decisions)
    constrained = []
    for budget in budgets:
        if not math.isfinite(budget) or not 0 <= budget <= 1:
            raise ValueError("FPR budgets must lie in [0, 1]")
        constrained.append(positive["violated"]
                           if negative["violated"] <= budget else None)
    return {"vsr": positive["violated"], "fpr": negative["violated"],
            "positive_rates": positive, "negative_rates": negative,
            "constrained_vsr": constrained,
            "metric_scope": "fixed-rule, per-fingerprint empirical rates"}
