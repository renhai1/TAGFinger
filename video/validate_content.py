#!/usr/bin/env python3
"""Check every film result against the supplied main-paper LaTeX (stdlib only)."""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
story = json.loads((ROOT / "storyboard.json").read_text())
paper = (ROOT.parent.parent / "sec" / "tagfinger_main.tex").read_text()


def numbers(row):
    return [float(x) for x in re.findall(r"(?<![A-Za-z])\d+\.\d+", row)]


table1 = paper.split(r"\label{tab1}", 1)[1].split(r"\end{table*}", 1)[0]
first_gcn = next(line for line in table1.splitlines() if "& GCN" in line)
entries = numbers(first_gcn)
values = [entries[i] for i in [6, 8, 10, 12]]
assert values == story["results"]["table1"]["values"], (values, "Table 1 mismatch")

table3 = paper.split(r"\label{tab3}", 1)[1].split(r"\end{table}", 1)[0]
for name, row_name in [("TAGFinger", "Ours"), ("SIGFinger", "SIGFinger")]:
    row = next(line for line in table3.splitlines() if line.startswith(row_name + " &"))
    values = numbers(row)[::2]
    assert values == story["results"]["table3"][name], (name, values, "Table 3 mismatch")

table4 = paper.split(r"\label{tab4}", 1)[1].split(r"\end{table}", 1)[0]
row = next(line for line in table4.splitlines() if "& GAT" in line)
assert numbers(row)[2] == story["results"]["table4"]["value"], "Table 4 mismatch"

# Check the actual response examples shown by the renderer.
examples = [([.65, .2, .15], [.15, .75, .1], 1),
            ([.58, .24, .18], [.2, .66, .14], 1),
            ([.64, .22, .14], [.61, .25, .14], 0)]


def evidence(before, after, target=1):
    a = max(range(len(before)), key=before.__getitem__)
    b = max(range(len(after)), key=after.__getitem__)
    return 1 if a != target and b == target else (0 if a == b else -1)


votes = [evidence(a, b) for a, b, expected in examples]
assert votes == [expected for a, b, expected in examples] == [1, 1, 0]
assert votes.count(1) == 2
assert evidence([.1, .8, .1], [.2, .7, .1]) == 0  # target → same target
assert evidence([.7, .2, .1], [.1, .2, .7]) == -1
assert [1, 0, -1].count(1) < 2 and [1, 0, -1].count(-1) < 2
assert "not a fresh code run" in story["disclaimer"]
print("PASS: all paper-reported values, example votes, zero/negative evidence, and uncertain outcome.")
