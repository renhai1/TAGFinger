"""Explicit provenance checks for a supplied no-use training source.

Metadata is a declaration, not a mathematical or cryptographic proof of
independence. Exact source/text overlaps are rejected; no shifted control is
created from the protected data.
"""
import hashlib


def text_digest(text):
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def validate_control_provenance(protected_texts, control_texts, metadata,
                                protected_source_ids, label_space_id):
    if not isinstance(metadata, dict):
        raise ValueError("Control provenance must be a JSON object")
    sources = metadata.get("training_source_ids")
    if not isinstance(sources, list) or not sources or any(
            not isinstance(s, str) or not s.strip() for s in sources):
        raise ValueError("Control provenance must declare training_source_ids")
    if set(sources) & set(protected_source_ids):
        raise ValueError("Control training sources overlap protected sources")
    if not label_space_id or metadata.get("label_space_id") != label_space_id:
        raise ValueError("Control and audited tasks must declare the same label_space_id")
    protected = {text_digest(t) for t in protected_texts if t.strip()}
    if any(text_digest(t) in protected for t in control_texts if t.strip()):
        raise ValueError("Control texts overlap the protected source")
    return {"training_source_ids": sources, "label_space_id": label_space_id,
            "provenance_scope": "declared sources with exact-text overlap checks"}
