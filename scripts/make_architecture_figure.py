"""Draw the GNN-IDS architecture diagram -> figures/architecture.svg and figures/architecture.png.

    python scripts/make_architecture_figure.py

Development-only helper (needs matplotlib, which is not a runtime dependency).
Dimensions and parameter counts are those of the implementation in gnn_ids/models/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

matplotlib.rcParams["svg.fonttype"] = "path"      # glyphs as outlines: identical in every viewer
matplotlib.rcParams["font.family"] = "DejaVu Sans"

INK = "#222222"
BLUE = ("#E9F1FB", "#3B6EA8")
ORANGE = ("#FDF1E1", "#C77D1A")
GREEN = ("#E8F4EA", "#3E8E50")
GRAY = ("#F3F3F3", "#7A7A7A")
WHITE = "#FFFFFF"

fig, ax = plt.subplots(figsize=(17.5, 9.4))
ax.set_xlim(0, 175)
ax.set_ylim(0, 94)
ax.set_aspect("equal")
ax.axis("off")


def panel(x, y, w, h, title, colors, sub=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.5",
                                fc=colors[0], ec=colors[1], lw=1.6))
    ax.text(x + w / 2, y + h - 2.4, title, ha="center", va="center", fontsize=11,
            fontweight="bold", color=colors[1])
    if sub:
        ax.text(x + w / 2, y + h - 5.4, sub, ha="center", va="center", fontsize=8.5,
                color=INK, style="italic")


def box(x, y, w, h, text, ec=INK, fc=WHITE, fs=9, lw=1.2, ls="-", weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8",
                                fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            linespacing=1.35, fontweight=weight)


def arrow(p, q, color=INK, lw=1.5, style="-|>", ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=13, lw=lw,
                                 color=color, ls=ls, shrinkA=0, shrinkB=0,
                                 connectionstyle=f"arc3,rad={rad}"))


def label(x, y, text, fs=8.5, ha="center", color=INK, style="normal"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color=color, style=style,
            linespacing=1.3)


# ----------------------------------------------------------------------------- title
ax.text(87.5, 91.0, "GNN-IDS: communication-graph GCN with post-GCN attention for flow-level "
        "intrusion detection", ha="center", va="center", fontsize=14, fontweight="bold",
        color=INK)
ax.text(87.5, 87.6, "10,180 trainable parameters = 4,288 (GCN) + 3,779 (attention) + "
        "2,113 (classifier)", ha="center", va="center", fontsize=10, color="#444444")

# ----------------------------------------------------------------- 1. training graph
panel(2, 44, 31, 38, "Training graph", BLUE, "built from training flows only")
box(4.5, 62, 26, 12, "Nodes: IP addresses seen in\ntraining flows + 1 shared\nplaceholder (unseen IPs)")
box(4.5, 55, 26, 5, "node features  x = 1   [N, 1]", fs=8.5)
box(4.5, 46.5, 26, 6.5, "edge_index: forward + reverse\ncopy of every flow, 2N edges,\nduplicates kept, no weights", fs=8.5)

# --------------------------------------------------------------------------- 2. GCN
panel(38, 44, 30, 38, "GCNEncoder", BLUE, "4,288 parameters")
box(41, 66.5, 24, 7, "GCNConv  1 → 64", fc=WHITE)
box(41, 59.5, 24, 4.5, "ReLU", fs=9)
box(41, 52, 24, 5.5, "GCNConv  64 → 64", fc=WHITE)
arrow((53, 66.5), (53, 64.0), lw=1.2)
arrow((53, 59.5), (53, 57.5), lw=1.2)
box(41, 46, 24, 4.5, "node embeddings  H  [N, 64]", ec=BLUE[1], fs=8.5)
arrow((53, 52), (53, 50.5), lw=1.2)
arrow((33, 63), (38, 63), color=BLUE[1], lw=2.0)

# -------------------------------------------------------------------- 3. attention
panel(73, 30, 63, 54, "Post-GCN attention (per flow)", ORANGE, "3,779 parameters; no edge_index, no labels")
box(76, 64, 21, 8, "$h_{src}$ = H[flow_src]\n[B, 64]", fs=8.5)
box(76, 53, 21, 8, "$h_{dst}$ = H[flow_dst]\n[B, 64]", fs=8.5)
box(76, 38, 21, 11, "$z_{edge}$ = ReLU(Linear\n49 → 64)(x_edge)\n[B, 64]", fs=8.5)
box(101, 51, 20, 17, "concat\n[B, 192]\n↓\nLinear 192 → 3\n↓\nsoftmax", ec=ORANGE[1], fs=8.5, lw=1.6)
box(124, 51, 10, 17, "z_flow\n= Σ α$_i$ s$_i$\n[B, 64]", ec=ORANGE[1], fs=8.5, lw=1.6)
# components -> concat
arrow((97, 68), (101, 63.5), lw=1.1)
arrow((97, 57), (101, 58.5), lw=1.1)
arrow((97, 43.5), (101, 53), lw=1.1)
# alpha into the weighted sum
arrow((121, 59.5), (124, 59.5), color=ORANGE[1], lw=1.8)
label(122.5, 62.3, "α", fs=10, color=ORANGE[1])
# components also act as the values s1, s2, s3 (routed above the scorer)
ax.plot([97, 97, 129], [70.5, 74, 74], color=INK, lw=1.1)
ax.plot([129, 129], [74, 68.4], color=INK, lw=1.1)
arrow((129, 70.5), (129, 68.3), lw=1.1)
label(113, 75.6, "$s_1, s_2, s_3$ = ($h_{src}$, $h_{dst}$, $z_{edge}$) are also the values that are summed", fs=8)
# gather arrows from the GCN
arrow((65, 48.5), (76, 66.5), color=BLUE[1], lw=1.8, rad=-0.18)
arrow((65, 48.5), (76, 56.5), color=BLUE[1], lw=1.8, rad=-0.05)
label(71.2, 42.4, "row lookup per flow", fs=8, style="italic", color=BLUE[1])

# ---------------------------------------------------------------- flow feature input
box(76, 20, 21, 7.5, "Flow features x_edge\n[B, 49]  (scaled NetFlow)", fc=GRAY[0], ec=GRAY[1], fs=8.5)
arrow((86.5, 27.5), (86.5, 38), color=GRAY[1], lw=1.6)

# ------------------------------------------------------------------ 4. classifier
panel(141, 44, 31, 38, "FlowClassifier", GREEN, "2,113 parameters")
box(144, 65, 25, 6.5, "Linear  64 → 32", fc=WHITE)
box(144, 57.5, 25, 4.5, "ReLU", fs=9)
box(144, 50, 25, 5.5, "Linear  32 → 1", fc=WHITE)
arrow((156.5, 65), (156.5, 62.0), lw=1.2)
arrow((156.5, 57.5), (156.5, 55.5), lw=1.2)
box(144, 45.5, 25, 3.2, "logit  →  sigmoid  →  p(attack)", ec=GREEN[1], fs=8, lw=1.4)
arrow((156.5, 50), (156.5, 48.7), lw=1.2)
arrow((134, 59.5), (141, 59.5), color=ORANGE[1], lw=2.0)
label(137.6, 62.3, "$z_{flow}$", fs=8.5)

# --------------------------------------------------------------- GCN-only ablation
box(141, 28, 31, 11.5, "GCN-only ablation (section 11)\n$z_{flow}$ = ½($h_{src}$ + $h_{dst}$)\nsame classifier, no flow features,\nno attention", ec=GRAY[1], fc=WHITE, fs=8.5, ls="--")
arrow((156.5, 44), (156.5, 39.5), color=GRAY[1], lw=1.2, ls="--", style="<|-|>")

# ----------------------------------------------------------- training / evaluation
box(2, 2, 84, 15.5,
    "\n"
    "1. one graph-wide GCN forward; the embeddings are detached into a leaf tensor\n"
    "2. mini-batches (8,192 flows): gather → attention → classifier →\n"
    "    BCEWithLogits(pos_weight, sum) / N;  backward into the leaf\n"
    "3. one backward through the GCN with the accumulated gradient;  Adam step (lr 1e-3)",
    ec=GRAY[1], fc="#FAFAFA", fs=8.6, lw=1.3)
box(90, 2, 82, 15.5,
    "\n"
    "Validation and test flows are never added to the graph: they look up the embeddings of\n"
    "their endpoint nodes (unseen IP → placeholder node).\n"
    "Checkpoint and decision threshold are chosen on validation only (PR-AUC selection,\n"
    "F1-maximising threshold, frozen); the test split is scored once.",
    ec=GRAY[1], fc="#FAFAFA", fs=8.6, lw=1.3)

ax.text(44, 15.3, "Training, per epoch", ha="center", va="center", fontsize=10,
        fontweight="bold", color=INK)
ax.text(131, 15.3, "Evaluation protocol", ha="center", va="center", fontsize=10,
        fontweight="bold", color=INK)

# ------------------------------------------------------------------------- output
out = Path(__file__).resolve().parents[1] / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "architecture.svg", format="svg", bbox_inches="tight", pad_inches=0.15)
fig.savefig(out / "architecture.png", format="png", dpi=300, bbox_inches="tight",
            pad_inches=0.15, facecolor="white")
print("wrote", out / "architecture.svg", "and", out / "architecture.png")
