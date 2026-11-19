#!/usr/bin/env python3
"""Reproducible, vector-style TAGFinger research film.

All method examples are explicitly schematic. Only storyboard.result values
are empirical, and each is tied to a specific table and experimental setting.
Pillow renders the original graphics; FFmpeg produces a portable H.264 MP4.
Piper is optional when reusing the provided narration.wav and timeline.json.
"""
from __future__ import annotations

import argparse
import functools
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
import wave

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
STORY = json.loads((ROOT / "storyboard.json").read_text())
BG = "#F5F3EC"
INK = "#172A35"
MUTED = "#6B7880"
LINE = "#DCDDD5"
WHITE = "#FFFFFF"
BLUE = "#237F9D"
BLUE_LIGHT = "#E4F0F4"
GREEN = "#128A70"
GREEN_LIGHT = "#E0F0E8"
GOLD = "#C69632"
GOLD_LIGHT = "#FAEECF"
CORAL = "#C96352"
CORAL_LIGHT = "#F8E7E0"
DARK = "#172E3A"
POSITIONS = [(.12, .27), (.38, .13), (.64, .26), (.87, .15),
             (.18, .77), (.49, .66), (.76, .78), (.89, .48)]
EDGES = [(0, 1), (0, 4), (0, 5), (1, 2), (1, 5), (2, 3),
         (2, 5), (2, 7), (3, 7), (4, 5), (5, 6), (6, 7)]
STABLE = {1, 2, 5}


def clamp(x, lo=0., hi=1.):
    return max(lo, min(hi, x))


def ease(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def blend(a, b, amount):
    from PIL import ImageColor
    aa, bb = ImageColor.getrgb(a), ImageColor.getrgb(b)
    return tuple(round(x + (y - x) * clamp(amount)) for x, y in zip(aa, bb))


@functools.lru_cache(maxsize=256)
def font(size, weight="regular", symbols=False):
    # Deliberately use ordinary font files, not screenshots of a paper.
    apple = Path("/System/Library/Fonts/Avenir Next.ttc")
    if symbols:
        symbol_path=Path('/System/Library/Fonts/Supplemental') / ('Arial Bold.ttf' if weight=='bold' else 'Arial.ttf')
        if symbol_path.exists():
            return ImageFont.truetype(str(symbol_path),size)
    if apple.exists():
        return ImageFont.truetype(str(apple), size, index=0 if weight == "bold" else 5)
    names = (["DejaVuSans-Bold.ttf", "Arial Bold.ttf"] if weight == "bold"
             else ["DejaVuSans.ttf", "Arial.ttf"])
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    raise RuntimeError("Install Avenir Next, Arial, or DejaVu Sans for rendering")


class Canvas:
    """Logical 1600 x 900 layout, rasterized at the requested output size."""
    def __init__(self, width=1920):
        self.s = width / 1600
        self.im = Image.new("RGB", (width, round(width * 9 / 16)), BG)
        self.d = ImageDraw.Draw(self.im)

    def box(self, x, y, w, h):
        return tuple(round(v * self.s) for v in (x, y, x+w, y+h))

    def rect(self, x, y, w, h, fill, radius=0, outline=None, thickness=1):
        args = dict(fill=fill, outline=outline, width=max(1, round(thickness*self.s)))
        if radius:
            self.d.rounded_rectangle(self.box(x, y, w, h), radius=round(radius*self.s), **args)
        else:
            self.d.rectangle(self.box(x, y, w, h), **args)

    def panel(self, x, y, w, h, fill=WHITE, outline=LINE):
        self.rect(x+1, y+5, w, h, "#E8E6DF", 18)
        self.rect(x, y, w, h, fill, 18, outline)

    def line(self, xy, color=LINE, width=2):
        self.d.line([(round(x*self.s), round(y*self.s)) for x, y in xy],
                    fill=color, width=max(1, round(width*self.s)), joint="curve")

    def circle(self, x, y, r, fill, outline=None, width=2):
        self.d.ellipse(self.box(x-r, y-r, r*2, r*2), fill=fill, outline=outline,
                       width=max(1, round(width*self.s)))

    def text(self, x, y, txt, size=24, color=INK, weight="regular", anchor="lt"):
        self.d.text((round(x*self.s), round(y*self.s)), str(txt),
                    font=font(round(size*self.s), weight, any(ch in str(txt) for ch in '→↔')), fill=color, anchor=anchor)

    def measure(self, txt, size=24, weight="regular"):
        return self.d.textlength(str(txt), font=font(round(size*self.s), weight, any(ch in str(txt) for ch in '→↔'))) / self.s

    def paragraph(self, x, y, txt, width, size=24, color=INK, weight="regular", leading=1.35):
        lines, current = [], ""
        for word in txt.split():
            trial = (current + " " + word).strip()
            if current and self.measure(trial, size, weight) > width:
                lines.append(current)
                current = word
            else:
                current = trial
        if current:
            lines.append(current)
        for i, line in enumerate(lines):
            self.text(x, y+i*size*leading, line, size, color, weight)
        return len(lines)*size*leading

    def pill(self, x, y, txt, color=BLUE, fill=BLUE_LIGHT, size=17):
        w = self.measure(txt, size, "bold") + 30
        self.rect(x, y, w, size+18, fill, 10)
        self.text(x+15, y+9, txt, size, color, "bold")
        return w

    def arrow(self, x1, y1, x2, y2, color=BLUE, width=3, progress=1):
        p = ease(progress)
        if p <= 0:
            return
        end = (x1+(x2-x1)*p, y1+(y2-y1)*p)
        self.line([(x1, y1), end], color, width)
        angle = math.atan2(y2-y1, x2-x1)
        self.line([(end[0]-11*math.cos(angle-.55), end[1]-11*math.sin(angle-.55)),
                   end,
                  (end[0]-11*math.cos(angle+.55), end[1]-11*math.sin(angle+.55))], color, width)

    def document(self, x, y, w=46, h=58, color=BLUE, fill=WHITE):
        self.rect(x, y, w, h, fill, 6, color, 2)
        self.rect(x+9, y+11, w-18, 5, color, 2)
        for i in range(3):
            self.line([(x+9, y+26+i*9), (x+w-9-(5 if i==2 else 0), y+26+i*9)],
                      blend(color, fill, .55), 2)

    def lock(self, x, y, color=WHITE, size=1):
        self.d.arc(self.box(x+10*size, y, 28*size, 34*size), 180, 360, fill=color,
                   width=max(1, round(3*size*self.s)))
        self.rect(x, y+17*size, 48*size, 35*size, color, 7*size)
        self.circle(x+24*size, y+30*size, 3*size, DARK)

    def graph(self, x, y, w, h, stable=False, text_nodes=True, phase=0.,
              quiet=False, edge_focus=False, tiny=False, nodes=None, exclude_edges=()):
        points = [(x+px*w, y+py*h) for px, py in POSITIONS]
        nodes = set(range(len(points))) if nodes is None else set(nodes)
        if stable:
            poly = [points[i] for i in (1, 2, 5)]
            cx = sum(p[0] for p in poly)/3
            cy = sum(p[1] for p in poly)/3
            expanded = [(cx+(xx-cx)*1.40, cy+(yy-cy)*1.40) for xx, yy in poly]
            self.d.polygon([(round(xx*self.s), round(yy*self.s)) for xx, yy in expanded],
                           fill=GREEN_LIGHT)
            self.line(expanded+[expanded[0]], blend(GREEN, BG, .6), 2)
        for a, b in EDGES:
            if a not in nodes or b not in nodes or (a,b) in exclude_edges or (b,a) in exclude_edges:
                continue
            color = GREEN if stable and a in STABLE and b in STABLE else "#A9BEC6"
            if quiet and not(a in STABLE and b in STABLE):
                color = "#D4DFE1"
            self.line([points[a], points[b]], color, 3 if stable and a in STABLE and b in STABLE else 2)
        if edge_focus:
            self.line([points[2], points[5]], GOLD, 7)
        for i, (xx, yy) in enumerate(points):
            if i not in nodes:
                continue
            selected = stable and i in STABLE
            r = 11 if tiny else 17
            if selected:
                self.circle(xx, yy, r+7+2*math.sin(phase*2), GREEN_LIGHT, GREEN, 1)
            color = GREEN if selected else BLUE
            if quiet and not selected:
                color = "#AFC7D0"
            self.circle(xx, yy, r, color, WHITE, 2)
            if not tiny:
                self.text(xx, yy+1, str(i+1), 15, WHITE, "bold", "mm")
            if text_nodes and i in (0, 2, 5, 6) and not tiny:
                self.document(xx+18, yy-43, 26, 34, GREEN if selected else BLUE)
        return points


def progress_for_cue(scene, local, index):
    cue = scene["timed_cues"][min(index, len(scene["timed_cues"])-1)]
    return ease((local-cue["local_start"])/1.3)


def base_frame(scene, index, local, width):
    c = Canvas(width)
    # Compact, original graph/fingerprint wordmark.
    for x, y in ((53, 42), (64, 28), (79, 43)):
        c.circle(x, y, 4, GREEN)
    c.line([(53, 42), (64, 28), (79, 43), (53, 42)], GREEN, 2)
    c.text(94, 24, "TAGFinger", 24, INK, "bold")
    c.text(260, 31, "GRAPH DATA OWNERSHIP", 12, MUTED, "bold")
    for n in range(len(STORY["scenes"])):
        c.rect(1044+n*38, 37, 27, 4, GREEN if n <= index else LINE, 2)
    c.text(54, 94, scene["chapter"], 15, BLUE, "bold")
    title_size = 49
    while c.measure(scene["title"], title_size, "bold") > 1480:
        title_size -= 1
    c.text(50, 130, scene["title"], title_size, INK, "bold")
    c.line([(54, 774), (1546, 774)], LINE, 1)
    mode = "PAPER-REPORTED RESULTS" if scene["id"] in ("results", "mixture") else "METHOD ILLUSTRATION · SCHEMATIC"
    c.text(54, 787, "ANONYMOUS RESEARCH OVERVIEW", 12, MUTED)
    c.text(800, 787, mode, 12, MUTED, anchor="mt")
    c.text(1546, 786, f"{index+1:02d} / {len(STORY['scenes']):02d}", 13, MUTED, anchor="rt")
    return c


def draw_question(c, s, t):
    c.panel(54, 230, 775, 490)
    c.pill(80, 252, "YOUR TEXT-ATTRIBUTE GRAPH", BLUE)
    points = c.graph(88, 320, 430, 290, phase=t)
    c.line([points[2], (525, 380)], BLUE, 2)
    c.rect(515, 330, 280, 250, BG, 12)
    c.text(537, 350, "A paper is a node.", 23, INK, "bold")
    c.text(537, 394, "TITLE", 13, BLUE, "bold")
    c.paragraph(537, 419, "Graph neural networks for semi-supervised classification", 235, 20)
    c.text(537, 492, "ABSTRACT", 13, BLUE, "bold")
    c.paragraph(537, 516, "We study learning on a network of linked documents.", 235, 18, MUTED)
    c.text(84, 656, "Nodes = text-bearing documents", 21, INK)
    c.text(84, 684, "Edges = relationships, such as citations", 19, MUTED)
    c.arrow(858, 465, 1040, 465, BLUE, progress=progress_for_cue(s,t,1))
    c.text(947, 409, "train?", 20, MUTED, anchor="mt")
    c.panel(1064, 300, 482, 340, DARK, DARK)
    c.lock(1276, 336, WHITE, .85)
    c.text(1305, 413, "SUSPECTED MODEL", 22, WHITE, "bold", "mt")
    c.text(1305, 467, "Training data: unknown", 24, "#CAE0E7", anchor="mt")
    c.text(1305, 505, "Parameters: hidden", 24, "#CAE0E7", anchor="mt")
    c.rect(1122, 558, 365, 50, "#294752", 10)
    c.text(1305, 583, "Only output probabilities are visible", 18, WHITE, anchor="mm")
    c.pill(1100, 674, "QUESTION THE MODEL, NOT ITS FILES", GREEN, GREEN_LIGHT, 15)


def draw_challenge(c, s, t):
    c.panel(54, 240, 398, 430)
    c.pill(80, 260, "HIDDEN DATA MIXTURE")
    c.graph(91, 319, 305, 160, tiny=True, text_nodes=False)
    c.rect(88, 528, 315, 30, LINE, 8)
    split = .2 + .6*(1-ease(clamp((t-1)/8)))
    c.rect(88, 528, 315*split, 30, BLUE, 8)
    c.text(88, 583, "Your source", 20, BLUE, "bold")
    c.text(88, 614, "+ unknown additional data", 20, MUTED)
    c.panel(486, 240, 440, 430)
    c.pill(512, 260, "DIFFERENT LEARNING TASKS", GOLD, GOLD_LIGHT)
    for i, (name, sub) in enumerate([("Node", "Classify a document"), ("Edge", "Predict a link"), ("Graph", "Classify a whole graph")]):
        y = 338+i*96
        c.circle(536, y+16, 16, GOLD_LIGHT)
        c.text(536, y+16, str(i+1), 16, GOLD, "bold", "mm")
        c.text(573, y, name, 25, INK, "bold")
        c.text(573, y+37, sub, 19, MUTED)
    c.panel(960, 240, 586, 430)
    c.pill(986, 260, "UNKNOWN ARCHITECTURE", GREEN, GREEN_LIGHT)
    for i, name in enumerate(["GCN", "GAT", "GIN", "RGCN"]):
        x, y = 1001+(i%2)*251, 355+(i//2)*111
        c.rect(x, y, 210, 76, BG, 13, LINE)
        c.text(x+105, y+39, name, 28, INK, "bold", "mm")
    c.text(1253, 611, "A different aggregation rule", 22, MUTED, anchor="mt")
    c.pill(448, 707, "DATASET-SPECIFIC + TASK-SPECIFIC SIGNALS CAN WEAKEN", CORAL, CORAL_LIGHT, 19)


def draw_principle(c, s, t):
    c.panel(54, 230, 638, 464)
    c.pill(80, 252, "RELEASED DATASET", BLUE)
    c.graph(127, 318, 441, 246, text_nodes=True)
    c.pill(230, 598, "NO PRE-EMBEDDED TRIGGER", BLUE, BLUE_LIGHT, 18)
    c.arrow(722, 456, 835, 456, GOLD, progress=progress_for_cue(s,t,0))
    c.text(777, 405, "copy", 18, MUTED, anchor="mt")
    c.panel(864, 230, 682, 464)
    c.pill(890, 252, "AUDIT QUERY COPY", GREEN, GREEN_LIGHT)
    c.graph(938, 318, 447, 246, stable=True, phase=t)
    c.pill(1035, 598, "LOCAL TEXT EDITS AT AUDIT TIME", GREEN, GREEN_LIGHT, 18)
    c.text(800, 724, "The protected release stays unchanged. The audit input does not.", 28, INK, "bold", "mt")


def probability_bars(c, x, y, values, color, w=180, h=90):
    for i, value in enumerate(values):
        bw = w/len(values)-12
        bx = x+i*w/len(values)
        c.rect(bx, y+h-h*value, bw, max(2,h*value), color, 4)
    c.line([(x-3,y+h+2),(x+w-10,y+h+2)], LINE, 1)


def draw_alignment(c, s, t):
    c.panel(54, 238, 373, 465)
    c.pill(80, 260, "ADVERSARIAL GRAPH GENERATOR", GOLD, GOLD_LIGHT, 13)
    c.graph(104, 326, 260, 215, tiny=True, text_nodes=False)
    c.paragraph(87, 580, "Expose disagreements near the decision boundary.", 297, 23)
    c.arrow(452, 419, 561, 341, GOLD, progress=progress_for_cue(s,t,0))
    c.arrow(452, 493, 561, 602, GOLD, progress=progress_for_cue(s,t,0))
    c.panel(585, 238, 510, 200, DARK, DARK)
    c.pill(607, 258, "SUSPECTED / FROZEN", WHITE, "#294752", 14)
    c.text(610, 329, "Black-box GNN", 27, WHITE, "bold")
    probability_bars(c, 872, 312, [.27,.76,.41,.18], "#8CC4D5", 170, 81)
    c.panel(585, 497, 510, 206)
    c.pill(607, 517, "SURROGATE / TRAINABLE", GREEN, GREEN_LIGHT, 14)
    c.text(610, 592, "Query-based proxy", 25, INK, "bold")
    p = progress_for_cue(s,t,1)
    start, finish = [.74,.31,.68,.48], [.28,.73,.42,.19]
    values = [a+(b-a)*p for a,b in zip(start,finish)]
    probability_bars(c, 872, 572, values, GREEN, 170, 84)
    c.arrow(1140, 335, 1140, 600, MUTED, 2)
    c.panel(1196, 297, 350, 345, BG)
    c.text(1221, 324, "TWO ALIGNMENT SIGNALS", 15, BLUE, "bold")
    c.text(1221, 377, "DAC", 31, INK, "bold")
    c.paragraph(1221, 423, "Match output distributions", 288, 22)
    c.text(1221, 499, "DVC", 31, INK, "bold")
    c.paragraph(1221, 546, "Match response changes", 288, 22)
    c.text(800, 725, "SAKA · sensitivity-guided adversarial knowledge alignment", 22, MUTED, anchor="mt")


def draw_region(c, s, t):
    c.panel(54, 230, 835, 482)
    c.pill(80, 252, "SURROGATE PREDICTIONS", BLUE)
    p = progress_for_cue(s,t,1)
    points=c.graph(109, 318, 661, 330, stable=p>.05, quiet=p>.4, phase=t)
    if p>.3:
        c.text(points[2][0]+30, points[2][1]+20, "target", 20, GREEN, "bold")
    c.panel(925, 230, 621, 482)
    c.pill(951, 252, "STRUCTURAL CONTEXT", GREEN, GREEN_LIGHT)
    c.text(961, 323, "Select consistent neighbors.", 29, INK, "bold")
    c.paragraph(961, 381, "Use predictive similarity in the one-hop neighborhood to identify a local stable region.", 533, 25)
    for i, (name, val) in enumerate([("Neighbor v2", .92), ("Neighbor v6", .86), ("Neighbor v8", .39)]):
        y=507+i*56
        c.text(961,y,name,20, GREEN if i<2 else MUTED)
        c.rect(1134,y+5,335,15,LINE,7)
        c.rect(1134,y+5,335*val*progress_for_cue(s,t,0),15,GREEN if i<2 else "#BAC6C8",7)
    c.text(961, 682, "Ranks shown are illustrative, not measured scores.", 15, MUTED)
    c.text(471, 735, "Target node + top-k stable neighbors", 22, GREEN, "bold", "mt")


def draw_editing(c, s, t):
    c.panel(54, 230, 734, 376)
    c.pill(80, 251, "NODE v3 / ILLUSTRATIVE TEXT EDIT", GREEN, GREEN_LIGHT, 15)
    c.text(87, 319, "ORIGINAL TEXT", 14, MUTED, "bold")
    c.paragraph(87, 351, "We study graph neural networks for semi-supervised classification.", 635, 30)
    c.line([(87,446),(754,446)],LINE,1)
    c.text(87, 473, "CANDIDATE EDIT", 14, GREEN, "bold")
    c.rect(87, 510, 636, 66, GREEN_LIGHT, 8)
    typed = "We investigate graph neural networks…"
    reveal = clamp((t-1.1)/2.2)
    c.text(105, 527, typed[:max(0,round(len(typed)*reveal))], 28, GREEN, "bold")
    if 0<reveal<1 and int(t*3)%2==0:
        cursor_x=105+c.measure(typed[:round(len(typed)*reveal)],28,"bold")
        c.line([(cursor_x+3,526),(cursor_x+3,557)],GREEN,2)
    c.text(80, 627, "Wording illustrates the operation; it is not a recorded optimized edit.", 16, MUTED)
    c.panel(827, 230, 719, 376)
    c.pill(853, 251, "CANDIDATE SEARCH ON THE SURROGATE", BLUE, BLUE_LIGHT, 15)
    c.rect(859, 314, 172, 93, DARK, 12)
    c.text(945, 343, "LOCAL LLM", 20, WHITE, "bold", "mt")
    c.text(945, 375, "text + structure", 15, "#CAE0E7", anchor="mt")
    c.arrow(1045, 360, 1120, 360, BLUE, progress=progress_for_cue(s,t,0))
    c.rect(1140, 314, 369, 93, BG, 12, LINE)
    c.text(1324, 340, "Same text encoder", 24, INK, "bold", "mt")
    c.text(1324, 375, "Exact re-encoding · bounded change", 17, MUTED, anchor="mt")
    for i, label in enumerate(["candidate A", "candidate B", "candidate C"]):
        y=454+i*43
        c.text(866, y, label, 20, GREEN if i==1 else MUTED)
        c.rect(1049,y+3,320,14,LINE,6)
        p=progress_for_cue(s,t,1)
        c.rect(1049,y+3,320*[.41,.85,.56][i]*p,14,GREEN if i==1 else "#A8BDC5",6)
        if i==1 and p>.8:
            c.text(1394,y-1,"SELECT",17,GREEN,"bold")
    c.panel(54, 675, 1492, 70, DARK, DARK)
    c.text(800, 710, "Maximize target response + regional drift  ↔  update prompt features", 26, WHITE, "bold", "mm")
    c.text(1183, 633, "No suspected-model queries here. No gradient through the LLM.", 16, MUTED, anchor="mt")


def draw_tasks(c, s, t):
    labels=[("NODE", "Node-centered subgraph", BLUE), ("EDGE", "Endpoint-centered subgraph", GOLD), ("GRAPH", "Original labeled graph", GREEN)]
    for i, (name, detail, color) in enumerate(labels):
        x=54+i*510
        c.panel(x,230,472,419)
        c.pill(x+25,250,name,color,blend(color,WHITE,.88))
        node_sets=[{1,2,3,5,7},{0,1,2,5,6,7},set(range(8))]
        pts=c.graph(x+60,328,270,196,text_nodes=False,tiny=True,
                    nodes=node_sets[i],exclude_edges=[(2,5)] if i==1 else ())
        if i==0:
            c.circle(pts[2][0],pts[2][1],20,None,BLUE,3)
        if i==1:
            for target in (2,5):
                c.circle(pts[target][0],pts[target][1],20,None,GOLD,3)
        p=progress_for_cue(s,t,1)
        if p>0:
            prompt_color=blend(CORAL,WHITE,1-p)
            for j,(xx,yy) in enumerate([(x+369,387),(x+402,430),(x+359,467)]):
                if j:
                    c.line([(x+369,387),(xx,yy)],prompt_color,2)
                c.rect(xx-9,yy-9,18,18,prompt_color,4)
            c.line([pts[7],(x+369,387)],prompt_color,2)
            c.text(x+357,515,"prompt graph",16,prompt_color,anchor="mt")
        c.text(x+236,566,detail,21,INK,"bold","mt")
        c.text(x+236,606,"Own task label space",18,MUTED,anchor="mt")
    c.rect(54,674,1492,60,DARK,13)
    c.text(800,704,"Shared graph/subgraph query formulation — not shared class labels",26,WHITE,"bold","mm")
    c.text(800,759,"Requires a compatible graph/subgraph classification interface.",16,MUTED,anchor="mb")


def response_row(c,y,name,before,after,target="B", kind=1, progress=1):
    c.text(112,y+44,name,24,INK,"bold","lm")
    for x,vals,label in [(433,before,"clean"),(791,after,"edited")]:
        c.text(x-26,y+11,label,14,MUTED)
        for i,(letter,v) in enumerate(zip("ABC",vals)):
            bx=x+i*70
            color=GOLD if letter==target else "#9CB5C0"
            height=76*v*(1 if label=="clean" else ease(progress))
            c.rect(bx,y+97-height,46,max(2,height),color,4)
            c.text(bx+23,y+107,letter,15,GOLD if letter==target else MUTED,"bold","mt")
        c.line([(x-3,y+99),(x+195,y+99)],LINE,1)
    c.arrow(670,y+70,744,y+70,BLUE,2,progress)
    color=GREEN if kind==1 else (CORAL if kind==-1 else MUTED)
    fill=GREEN_LIGHT if kind==1 else (CORAL_LIGHT if kind==-1 else BG)
    txt="+1 · toward target" if kind==1 else ("−1 · other change" if kind==-1 else "0 · unchanged")
    c.rect(1107,y+32,362,57,fill,12)
    if progress>.5:
        c.text(1288,y+62,txt,22,color,"bold","mm")


def draw_evidence(c,s,t):
    c.panel(54,225,1492,446)
    c.pill(78,244,"SCHEMATIC PROBABILITIES · TASK-LOCAL TARGETS",GOLD,GOLD_LIGHT,14)
    c.text(570,259,"Clean query",19,MUTED,"bold","mt")
    c.text(913,259,"Edited query",19,MUTED,"bold","mt")
    c.text(1288,259,"Evidence",19,MUTED,"bold","mt")
    values=[("Node",[.65,.2,.15],[.15,.75,.1],1),
            ("Edge",[.58,.24,.18],[.2,.66,.14],1),
            ("Graph",[.64,.22,.14],[.61,.25,.14],0)]
    for i,(name,before,after,kind) in enumerate(values):
        p=ease((t-s['timed_cues'][1]['local_start']-i*.75)/1.4)
        response_row(c,292+i*118,name,before,after,kind=kind,progress=p)
    c.rect(54,699,1492,58,GREEN_LIGHT,12)
    p=progress_for_cue(s,t,2)
    label="Compare clean and edited predictions, task by task."
    if p>.15:
        label="2 positive votes ≥ threshold 2   →   support the dataset-use decision"
    c.text(800,729,label,26,GREEN,"bold","mm")


def draw_uncertainty(c,s,t):
    tiles=[("+1", "Toward the target", "Clean is non-target; edited is target.", GREEN,GREEN_LIGHT),
           ("0", "No prediction change", "Includes target → same target.", MUTED,BG),
           ("−1", "Another prediction change", "Changed, but not toward the target.", CORAL,CORAL_LIGHT)]
    for i,(value,title,detail,color,fill) in enumerate(tiles):
        x=54+i*510
        c.panel(x,230,472,320)
        c.circle(x+72,295,35,fill)
        c.text(x+72,295,value,32,color,"bold","mm")
        c.paragraph(x+27,358,title,425,27,INK,"bold")
        c.paragraph(x+27,433,detail,408,23,MUTED)
    c.panel(54,584,820,155,DARK,DARK)
    c.text(83,606,"ILLUSTRATIVE VOTES:  +1, 0, −1",21,"#CAE0E7","bold")
    c.text(83,649,"Neither sign reaches two votes.",27,WHITE,"bold")
    c.pill(549,643,"UNCERTAIN",GOLD,GOLD_LIGHT,21)
    c.panel(906,584,640,155)
    c.text(934,607,"CHECK FALSE POSITIVES",16,BLUE,"bold")
    c.paragraph(934,645,"Evaluate with genuine no-use controls, not only models trained on the protected data.",574,21)


def draw_results(c,s,t):
    data=STORY['results']['table1']
    c.panel(54,230,1006,516)
    c.pill(80,251,"CORA-ML / GCN / NODE TASK / S@1",BLUE,BLUE_LIGHT,16)
    left,right,top,bottom=156,1005,348,654
    for val in (0,25,50,75,100):
        y=bottom-(bottom-top)*val/100
        c.line([(left,y),(right,y)],LINE,1)
        c.text(left-22,y,f"{val}",15,MUTED,anchor="rm")
    c.text(84,315,"VSR (%)",16,MUTED)
    p=ease((t-.8)/3.5)
    for i,(name,v) in enumerate(zip(data['methods'],data['values'])):
        x=204+i*211
        color=GREEN if name=="TAGFinger" else "#9AB2BC"
        height=(bottom-top)*v/100*p
        c.rect(x,bottom-height,112,max(2,height),color,8)
        c.text(x+56,bottom-height-16,f"{v:.1f}",29,GREEN if i==3 else INK,"bold","mb")
        c.text(x+56,bottom+21,name,20,INK,"bold" if i==3 else "regular","mt")
    c.text(84,718,"Source: submitted main paper, Table 1. Selected non-intrusive baselines.",15,MUTED)
    c.panel(1094,230,452,516,DARK,DARK)
    c.text(1320,277,"TAGFinger",30,WHITE,"bold","mt")
    c.text(1320,340,"95.4%",77,"#8CDEC0","bold","mt")
    c.text(1320,442,"S@1",38,WHITE,"bold","mt")
    c.paragraph(1132,511,"Verification success rate at FPR ≤ 1%",377,27,WHITE,"bold")
    c.paragraph(1132,625,"Paper-reported values. Not a fresh run of the revised implementation.",375,20,"#CAE0E7")


def draw_mixture(c,s,t):
    data=STORY['results']['table3']
    c.panel(54,230,939,516)
    c.pill(80,251,"CORA-ML + REDDIT / NODE TASK / S@1",BLUE,BLUE_LIGHT,15)
    left,right,top,bottom=157,905,350,633
    for v in (0,25,50,75,100):
        y=bottom-(bottom-top)*v/100
        c.line([(left,y),(right,y)],LINE,1)
        c.text(left-21,y,str(v),15,MUTED,anchor="rm")
    c.text(80,316,"VSR (%)",16,MUTED)
    p=ease((t-.5)/3.5)
    for name,color in [("SIGFinger","#9BB0B8"),("TAGFinger",GREEN)]:
        pts=[]
        for i,v in enumerate(data[name]):
            x=left+i*(right-left)/3
            y=bottom-(bottom-top)*v/100
            pts.append((x,y))
        for i in range(3):
            c.line([pts[i],pts[i+1]],blend(color,WHITE,1-p),4)
        for i,(x,y) in enumerate(pts):
            c.circle(x,y,7,blend(color,WHITE,1-p),WHITE,2)
            if p>.5:
                c.text(x,y-16,f"{data[name][i]:.1f}",24,color,"bold","mb")
    for i,v in enumerate(data['owner_percentages']):
        c.text(left+i*(right-left)/3,bottom+21,f"{v}%",21,INK,"bold","mt")
    c.text(531,688,"Cora-ML share of the training mixture",21,MUTED,anchor="mt")
    c.text(81,719,"Source: main paper, Table 3. Comparator: SIGFinger.",15,MUTED)
    c.circle(551,317,5,GREEN);c.text(565,308,"TAGFinger",17,GREEN,"bold")
    c.circle(751,317,5,"#9BB0B8");c.text(765,308,"SIGFinger",17,MUTED)
    c.panel(1027,230,519,239,GREEN_LIGHT)
    c.text(1055,257,"20% CORA-ML + 80% REDDIT",16,GREEN,"bold")
    c.text(1055,311,"91.1%",64,GREEN,"bold")
    c.text(1055,402,"S@1 · main paper, Table 3",20,INK)
    c.panel(1027,499,519,247,DARK,DARK)
    c.text(1055,526,"CROSS-ARCHITECTURE / REDDIT",16,"#A9D6E4","bold")
    c.text(1055,573,"GCN → GAT",31,WHITE,"bold")
    c.text(1055,622,"93.5%",54,"#8CDEC0","bold")
    c.text(1055,707,"S@1 · main paper, Table 4",19,"#CAE0E7")


def draw_closing(c,s,t):
    c.text(800,251,"TAGFinger",87,INK,"bold","mt")
    c.text(800,359,"From local text edits to ownership evidence.",31,GREEN,"bold","mt")
    for i,(num,title,detail,color) in enumerate([
        ("01","QUERY-BASED PROXY","Align predictions and sensitivity",BLUE),
        ("02","SEMANTIC FINGERPRINT","Anchor edits in stable regions",GREEN),
        ("03","TASK-WISE EVIDENCE","Aggregate; retain uncertainty",GOLD)]):
        x=54+i*510
        c.panel(x,446,472,196)
        c.text(x+27,470,num,30,color,"bold")
        c.text(x+27,525,title,20,INK,"bold")
        c.text(x+27,579,detail,20,MUTED)
    c.text(800,687,"Semantic-Prompted Structural-Resilient Fingerprinting",23,INK,anchor="mt")
    c.text(800,724,"for Universal Ownership Verification of Text-Attribute Graphs",22,MUTED,anchor="mt")


DRAWERS={"question":draw_question,"challenge":draw_challenge,"principle":draw_principle,
         "alignment":draw_alignment,"region":draw_region,"editing":draw_editing,
         "tasks":draw_tasks,"evidence":draw_evidence,"uncertainty":draw_uncertainty,
         "results":draw_results,"mixture":draw_mixture,"closing":draw_closing}


def caption_text(text):
    for a,b in [("TAG Finger","TAGFinger"),("Cora M L","Cora-ML"),
                ("G C N","GCN"),("G A T","GAT"),("S at one","S@1"),
                ("ninety-five point four percent","95.4%"),
                ("ninety-one point one percent","91.1%"),
                ("ninety-three point five percent","93.5%")]:
        text=text.replace(a,b)
    return text


def frame(scene,index,local,width=1920,captions=True):
    c=base_frame(scene,index,local,width)
    DRAWERS[scene['id']](c,scene,local)
    # Soft scene entrances without long blank screens or animated filler.
    duration=scene['end']-scene['start']
    opacity=min(ease(local/.38),ease((duration-local)/.32))
    if opacity<1:
        c.im=Image.blend(Image.new('RGB',c.im.size,BG),c.im,opacity)
        c.d=ImageDraw.Draw(c.im)
    if captions:
        text=""
        for cue in scene['timed_cues']:
            if cue['local_start']-.04<=local<=cue['local_end']+.08:
                text=cue['caption'];break
        if text:
            words=text.split();lines=[];line=""
            for word in words:
                candidate=(line+" "+word).strip()
                if line and c.measure(candidate,24)>1420:
                    lines.append(line);line=word
                else:line=candidate
            lines.append(line)
            if len(lines)>2:
                raise ValueError(f"Caption needs more than two lines: {text}")
            for n,line in enumerate(lines):
                c.text(800,827+n*31-(9 if len(lines)==2 else 0),line,24,INK,anchor="mt")
    return c.im


def timecode(t,sep=','):
    ms=round(t*1000);h,ms=divmod(ms,3600000);m,ms=divmod(ms,60000);s,ms=divmod(ms,1000)
    return f'{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}'


def prepare_audio(args):
    import numpy as np
    from piper import PiperVoice
    from piper.config import SynthesisConfig
    args.assets.mkdir(parents=True,exist_ok=True)
    voice=PiperVoice.load(args.voice_model,config_path=args.voice_config)
    config=SynthesisConfig(length_scale=args.length_scale,noise_scale=.5,noise_w_scale=.6)
    sample_rate=voice.config.sample_rate
    samples=[];cursor=0;timeline=[];subtitles=[]
    def silence(seconds):
        nonlocal cursor
        block=np.zeros(round(seconds*sample_rate),dtype=np.int16)
        samples.append(block);cursor+=len(block)
    for si,story_scene in enumerate(STORY['scenes']):
        scene=dict(story_scene);scene['start']=cursor/sample_rate;scene['timed_cues']=[]
        silence(.58)
        for ci,text in enumerate(scene['cues']):
            path=args.assets/f'voice_{si:02d}_{ci:02d}.wav'
            if not path.exists() or args.refresh_audio:
                with wave.open(str(path),'wb') as wav:
                    voice.synthesize_wav(text,wav,syn_config=config)
            with wave.open(str(path),'rb') as wav:
                if wav.getframerate()!=sample_rate or wav.getnchannels()!=1 or wav.getsampwidth()!=2:
                    raise ValueError('Unexpected narration format')
                pcm=np.frombuffer(wav.readframes(wav.getnframes()),dtype=np.int16).copy()
            active=np.where(np.abs(pcm.astype(np.int32))>160)[0]
            if len(active)==0:raise ValueError(f'Empty narration: {path}')
            margin=round(.065*sample_rate)
            pcm=pcm[max(0,active[0]-margin):min(len(pcm),active[-1]+margin)]
            start=cursor/sample_rate;samples.append(pcm);cursor+=len(pcm);end=cursor/sample_rate
            cue={'text':text,'caption':caption_text(text),'start':start,'end':end,
                 'local_start':start-scene['start'],'local_end':end-scene['start']}
            scene['timed_cues'].append(cue);subtitles.append(cue)
            silence(.22 if ci<len(scene['cues'])-1 else .7)
        scene['end']=cursor/sample_rate;timeline.append(scene)
        print(f'{si+1:02d} {scene["id"]:<12} {scene["end"]-scene["start"]:5.1f}s',flush=True)
    silence(.4);timeline[-1]['end']=cursor/sample_rate
    pcm=np.concatenate(samples)
    with wave.open(str(args.assets/'narration.wav'),'wb') as wav:
        wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(sample_rate);wav.writeframes(pcm.tobytes())
    metadata={'duration':len(pcm)/sample_rate,'sample_rate':sample_rate,'voice':STORY['voice'],
              'length_scale':args.length_scale,'disclaimer':STORY['disclaimer'],'scenes':timeline}
    (args.assets/'timeline.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    srt='\n\n'.join(f'{i+1}\n{timecode(c["start"])} --> {timecode(c["end"])}\n{c["caption"]}' for i,c in enumerate(subtitles))+'\n'
    (args.assets/'TAGFinger_Overview_v2.srt').write_text(srt)
    (args.assets/'TAGFinger_Overview_v2.vtt').write_text('WEBVTT\n\n'+'\n\n'.join(
        f'{timecode(c["start"],".")} --> {timecode(c["end"],".")}\n{c["caption"]}' for c in subtitles)+'\n')
    print(f'Total: {len(pcm)/sample_rate:.2f}s',flush=True)


def load_timeline(args):
    timeline=json.loads((args.assets/'timeline.json').read_text())
    if len(timeline['scenes'])!=len(STORY['scenes']):
        raise ValueError('Scene count changed: regenerate audio and timeline first.')
    for scene,source in zip(timeline['scenes'],STORY['scenes']):
        if scene['id']!=source['id'] or scene['cues']!=source['cues']:
            raise ValueError('Narration script changed: regenerate audio and timeline first.')
        scene['title']=source['title']
        scene['chapter']=source['chapter']
    return timeline


def make_stills(args):
    timeline=load_timeline(args)
    directory=args.assets/'stills';directory.mkdir(exist_ok=True)
    contact=Image.new('RGB',(1600,3*324),BG)
    d=ImageDraw.Draw(contact)
    for i,scene in enumerate(timeline['scenes']):
        duration=scene['end']-scene['start']
        im=frame(scene,i,duration*.83,args.width,False)
        im.save(directory/f'{i+1:02d}_{scene["id"]}.png')
        thumbnail=im.resize((400,225),Image.Resampling.LANCZOS)
        x=(i%4)*400;y=(i//4)*324
        contact.paste(thumbnail,(x,y))
        d.text((x+12,y+239),f'{i+1:02d}  {scene["id"]}',font=font(19,'bold'),fill=INK)
        d.text((x+12,y+272),f'{scene["start"]:.1f} – {scene["end"]:.1f}s',font=font(16),fill=MUTED)
    contact.save(args.assets/'storyboard_contact.png')
    scene=timeline['scenes'][0]
    poster=frame(scene,0,(scene['end']-scene['start'])*.8,args.width,False)
    poster.save(args.assets/'TAGFinger_Overview_v2_poster.jpg',quality=95)
    poster.save(ROOT.parent/'images'/'TAGFinger_Overview_v2_poster.jpg',quality=95)
    print(f'Stills: {directory}')


def render(args):
    timeline=load_timeline(args);duration=timeline['duration'];scenes=timeline['scenes']
    if args.scene:
        index=next(i for i,s in enumerate(scenes) if s['id']==args.scene)
        selected=scenes[index];offset=selected['start'];duration=selected['end']-offset
    else:offset=0
    if args.seconds:duration=min(duration,args.seconds)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    ffmpeg=args.ffmpeg or shutil.which('ffmpeg')
    if not ffmpeg:raise RuntimeError('FFmpeg is required; use --ffmpeg /path/to/ffmpeg')
    cmd=[ffmpeg,'-y','-hide_banner','-loglevel','warning','-f','rawvideo',
         '-pixel_format','rgb24','-video_size',f'{args.width}x{round(args.width*9/16)}',
         '-framerate',str(args.fps),'-i','pipe:0']
    cmd+=['-ss',str(offset),'-i',str(args.assets/'narration.wav')]
    cmd+=['-map','0:v:0','-map','1:a:0','-c:v','libx264','-preset','medium',
          '-crf','19','-pix_fmt','yuv420p','-c:a','aac','-b:a','160k','-ar','48000',
          '-af','loudnorm=I=-16:TP=-1.5:LRA=7','-t',str(duration),
          '-movflags','+faststart','-metadata','title='+STORY['title'],
          '-metadata:s:a:0','language=eng',
          '-metadata','comment=Anonymous research overview. Method diagrams are schematic; result values are paper-reported.',
          str(args.output)]
    frames=math.ceil(duration*args.fps);started=time.monotonic();index=0
    with (args.assets/'render.log').open('w') as log:
        process=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
        try:
            for n in range(frames):
                absolute=offset+n/args.fps
                while index<len(scenes)-1 and absolute>=scenes[index]['end']:
                    index+=1
                scene=scenes[index]
                im=frame(scene,index,absolute-scene['start'],args.width,not args.no_captions)
                process.stdin.write(im.tobytes())
                if n%max(1,args.fps*10)==0:
                    elapsed=time.monotonic()-started
                    print(f'{n/args.fps:6.1f}/{duration:.1f}s  {scene["id"]:<12}  {n/max(.01,elapsed):.1f} fps rendered',flush=True)
            process.stdin.close()
            code=process.wait()
            if code:raise RuntimeError(f'FFmpeg exited {code}; see {args.assets}/render.log')
        except BaseException:
            process.kill();process.wait();raise
    print(f'Rendered {args.output} in {time.monotonic()-started:.1f}s',flush=True)


def validate(args):
    timeline=load_timeline(args)
    errors=[];end=0
    for i,scene in enumerate(timeline['scenes']):
        if abs(scene['start']-end)>.001:errors.append('Non-contiguous scenes')
        for cue in scene['timed_cues']:
            if not(scene['start']<=cue['start']<cue['end']<=scene['end']):errors.append('Invalid cue times')
        for fraction in (.1,.4,.7,.9):
            frame(scene,i,(scene['end']-scene['start'])*fraction,args.width)
        end=scene['end']
    with wave.open(str(args.assets/'narration.wav'),'rb') as wav:
        if abs(wav.getnframes()/wav.getframerate()-end)>.01:errors.append('Audio/timeline duration mismatch')
    if errors:raise ValueError(errors)
    print(f'PASS: {len(timeline["scenes"])} scenes, all cue timings, caption lengths, audio duration ({end:.2f}s).')


def make_preview(args):
    timeline=load_timeline(args)
    editing=next(s for s in timeline['scenes'] if s['id']=='editing')
    evidence=next(s for s in timeline['scenes'] if s['id']=='evidence')
    start_a=editing['start']+1
    start_b=evidence['timed_cues'][1]['start']-.5
    ffmpeg=args.ffmpeg or shutil.which('ffmpeg')
    if not ffmpeg:raise RuntimeError('FFmpeg is required for the GIF preview')
    dest=ROOT.parent/'images'/'overview_v2.gif'
    filters=(f'[0:v]split=2[a0][b0];[a0]trim=start={start_a}:duration=6,setpts=PTS-STARTPTS[a];'
             f'[b0]trim=start={start_b}:duration=8,setpts=PTS-STARTPTS[b];'
             '[a][b]concat=n=2:v=1:a=0,fps=8,scale=960:-1:flags=lanczos,split[pi][vi];'
             '[pi]palettegen=max_colors=96:stats_mode=diff[p];'
             '[vi][p]paletteuse=dither=bayer:bayer_scale=4[out]')
    subprocess.run([ffmpeg,'-y','-hide_banner','-loglevel','warning','-i',str(args.output),
                    '-filter_complex',filters,'-map','[out]','-loop','0',str(dest)],check=True)
    print(f'Preview: {dest}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('audio','stills','render','validate','preview'))
    p.add_argument('--assets',type=Path,default=ROOT/'assets')
    p.add_argument('--output',type=Path,default=ROOT.parent/'images'/'TAGFinger_Overview_v2.mp4')
    p.add_argument('--voice-model');p.add_argument('--voice-config')
    p.add_argument('--length-scale',type=float,default=.82)
    p.add_argument('--refresh-audio',action='store_true')
    p.add_argument('--width',type=int,default=1920);p.add_argument('--fps',type=int,default=24)
    p.add_argument('--scene');p.add_argument('--seconds',type=float)
    p.add_argument('--no-captions',action='store_true');p.add_argument('--ffmpeg')
    args=p.parse_args()
    if args.mode=='audio' and not args.voice_model:p.error('--voice-model is required for audio generation')
    {'audio':prepare_audio,'stills':make_stills,'render':render,'validate':validate,'preview':make_preview}[args.mode](args)


if __name__=='__main__':
    main()
