"""Strict parsing of node-text edits (usable without model dependencies)."""
import re


def parse_edits(output, target_node=None):
    """Accept only explicit Node <id>: <text> records, never whole responses."""
    output = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL)
    if "<think>" in output:
        return {}  # An unfinished reasoning section is not an edited text.
    edits = {}
    for line in output.splitlines():
        match = re.fullmatch(r"\s*Node\s+(\d+)\s*[:：]\s*(\S.*)\s*", line)
        if match:
            node = int(match.group(1))
            if node in edits:
                return {}  # Ambiguous duplicate edits are rejected.
            edits[node] = match.group(2).strip()
    return edits
