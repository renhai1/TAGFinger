"""Prompt templates for LLM-based fingerprint perturbation generation.

The user prompt templates are transcribed from Appendix C (Figures 9 and 10)
of the paper, for single-source and multi-source dataset ownership
verification. The user content conveys (i) the textual attributes X_i of the
target node and its stable neighbors, (ii) the structural description
Gamma(G_i) of the induced subgraph, and (iii) the target label y* specified
by the fingerprint objective, i.e., delta_i ~ P_psi(delta | X_i, Gamma(G_i),
y*) (Eq. 6).
"""

USER_PROMPT_SINGLE = (
    "In the context of graph neural network ownership verification, models "
    "may exhibit latent behavioral signatures learned from a specific "
    "training dataset.\n"
    "Your task is to assist in constructing a subtle semantic perturbation "
    "that can serve as a reliable fingerprint for single-source "
    "verification.\n"
    "You should focus on preserving the original semantic meaning of the "
    "node attributes while introducing a minimally invasive yet detectable "
    "modification.\n"
    "The perturbation should be localized within a stable subgraph region "
    "and induce a consistent shift in the model's classification "
    "probability vectors without altering graph semantics."
)

USER_PROMPT_MULTI = (
    "In the context of graph neural network ownership verification under "
    "multi-source training conditions, a suspected model may be trained on "
    "a mixture of datasets with unknown proportions.\n"
    "Your role is to generate a semantically coherent perturbation that "
    "remains effective even when the model has only partially or implicitly "
    "learned the target dataset.\n"
    "The perturbation should be robust to dataset heterogeneity, preserve "
    "semantic plausibility, and induce a transferable distribution drift in "
    "the classification probability vectors within a stable subgraph region."
)

USER_CONTENT_TEMPLATE = (
    "Target node {node_id} with textual attributes:\n{target_text}\n\n"
    "Stable neighbor nodes with textual attributes:\n{neighbor_texts}\n\n"
    "Subgraph structure description:\n{structure}\n\n"
    "Target label specified by the fingerprint objective: {target_label}\n\n"
    "Propose edits to the texts of nodes in this stable region so that the "
    "classification probability distribution drifts toward the target label "
    "while the edited texts remain fluent and semantically plausible.\n"
    "Return each edit on its own line in the format 'Node <id>: <edited text>'."
)


def build_pasp_prompt(node_id, target_text, neighbor_texts, structure,
                      target_label, mode="single"):
    """Assemble the user prompt and user content of Appendix C."""
    system = USER_PROMPT_MULTI if mode == "multi" else USER_PROMPT_SINGLE
    neighbors = "\n".join(
        f"- Node {nid}: {text}" for nid, text in neighbor_texts
    )
    content = USER_CONTENT_TEMPLATE.format(
        node_id=node_id, target_text=target_text,
        neighbor_texts=neighbors, structure=structure,
        target_label=target_label)
    return f"{system}\n\n{content}"


# Auxiliary label-preserving templates used to build fine-tuning data.
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