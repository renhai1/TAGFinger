

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
