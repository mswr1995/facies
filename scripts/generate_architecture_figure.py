import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


OUT = "ubmk-2026/figs/architecture.png"


def box(ax, xy, wh, text, face, fontsize=16, weight="normal"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.6,
        edgecolor="black",
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        linespacing=1.45,
    )
    return patch


def arrow(ax, start, end, label=None, label_offset=(0, 0), size=18):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="-|>", lw=2.2, color="black", shrinkA=0, shrinkB=0),
    )
    if label:
        lx = (start[0] + end[0]) / 2 + label_offset[0]
        ly = (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(lx, ly, label, ha="center", va="center", fontsize=size)


fig, ax = plt.subplots(figsize=(9, 12), dpi=180)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

left_x = 0.07
left_w = 0.46
head_h = 0.095
leaf_x = 0.78
leaf_w = 0.17
leaf_h = 0.055

box(ax, (0.13, 0.89), (0.34, 0.075), "Grain Patch 96x96x3\n(binary mask applied)", "#f5f5f5", 15)
box(ax, (0.08, 0.76), (0.44, 0.075), "ResNet-18 Backbone (shared)\n$h_i \\in \\mathbb{R}^{512}$", "#f5f5f5", 15)
box(ax, (left_x, 0.58), (left_w, head_h), "$MLP^{(1)}$: Stage 1\nPeloid vs. Non-Peloid\n$\\alpha=0.25,\\ \\gamma=2.0$", "#eeeeff", 15)
box(ax, (left_x, 0.38), (left_w, head_h), "$MLP^{(2)}$: Stage 2\nOoid-like vs. Intraclast\n$\\alpha=0.50,\\ \\gamma=2.0$", "#eeeeff", 15)
box(ax, (left_x, 0.18), (left_w, head_h), "$MLP^{(3)}$: Stage 3\nIntact vs. Broken Ooid\n$\\alpha=0.75,\\ \\gamma=2.0$", "#eeeeff", 15)

box(ax, (leaf_x, 0.60), (leaf_w, leaf_h), "Peloid", "#d9f6d6", 15, "bold")
box(ax, (leaf_x, 0.40), (leaf_w, leaf_h), "Intraclast", "#d9f6d6", 14, "bold")
box(ax, (leaf_x, 0.21), (leaf_w, leaf_h), "Broken\nOoid", "#d9f6d6", 13, "bold")
box(ax, (leaf_x, 0.055), (leaf_w, leaf_h), "Ooid", "#d9f6d6", 15, "bold")

arrow(ax, (0.30, 0.89), (0.30, 0.835))
arrow(ax, (0.30, 0.76), (0.30, 0.675))
arrow(ax, (0.53, 0.627), (leaf_x, 0.627), "$p^{(1)}>0.5$", (0.0, 0.03), 15)
arrow(ax, (0.30, 0.58), (0.30, 0.475), "$p^{(1)}\\leq0.5$", (0.08, 0.0))

arrow(ax, (0.53, 0.427), (leaf_x, 0.427), "$p^{(2)}<0.5$", (0.0, 0.03), 15)
arrow(ax, (0.30, 0.38), (0.30, 0.275), "$p^{(2)}\\geq0.5$", (0.08, 0.0))

arrow(ax, (0.53, 0.227), (leaf_x, 0.237), "$p^{(3)}<0.5$", (0.0, 0.03), 15)
arrow(ax, (0.30, 0.18), (0.30, 0.082))
arrow(ax, (0.30, 0.082), (leaf_x, 0.082), "$p^{(3)}\\geq0.5$", (0.10, 0.03), 15)

plt.savefig(OUT, bbox_inches="tight", pad_inches=0.25)
plt.close(fig)
