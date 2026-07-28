"""Prompt templates for LLM-based fingerprint perturbation generation.

PASP_PROMPT conditions the LLM on (i) the textual attributes X_i of the
stable subgraph G_i, (ii) the structural description Gamma(G_i), and
(iii) the target label y* conveyed through the user-defined fingerprint
objective, i.e., delta_i ~ P_psi(delta | X_i, Gamma(G_i), y*) (Eq. 6).
"""

PASP_PROMPT = (
    "You are generating a prompt-amplified stable perturbation (fingerprint) "
    "for a text-attribute graph.\n\n"
    "Textual attributes of the stable subgraph nodes:\n{texts}\n\n"
    "Structural description of the subgraph:\n{structure}\n\n"
    "Fingerprint objective: rewrite the text of the target node with a subtle "
    "and natural-looking perturbation so that the classification probability "
    "distribution of a model trained on this dataset drifts toward the target "
    "label \"{target_label}\", while the perturbed text remains fluent, "
    "semantically plausible, and indistinguishable from normal data.\n\n"
    "Return only the perturbed text of the target node."
)


# Auxiliary label-preserving templates used for ablation studies.
PERTURB_PROMPTS = {
    "lexical": (
        "Generate a lexically perturbed version of the following text. "
        "You may replace words with synonyms, but the meaning and task label "
        "must remain unchanged.\n\nText:\n{text}"
    ),

    "syntactic": (
        "Generate a syntactically perturbed version of the following text. "
        "You may change sentence structure or grammar, but preserve meaning "
        "and task label.\n\nText:\n{text}"
    ),

    "discourse": (
        "Generate a discourse-level perturbed version of the following text. "
        "You may rewrite the text with a different style or narrative flow, "
        "but preserve semantics and task label.\n\nText:\n{text}"
    )
}
