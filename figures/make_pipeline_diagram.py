# This scripts draw the six-step policy engine pipeline as boxes and arrows for reports

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

steps = ["perceive", "reuse", "assign_direct", "generate_candidates", "select", "reconcile"]

fontsize = 13
char_w = 0.145      
gap = 0.45          
box_h = 0.5
y_center = 0.5

box_widths = [max(1.4, len(s) * char_w) for s in steps]

fig, ax = plt.subplots(figsize=(14, 2.6))

x = 0.0
centers = []
for w in box_widths:
    x += gap / 2
    centers.append(x + w / 2)
    x += w + gap / 2

ax.set_xlim(0, x)
ax.set_ylim(0, 1)
ax.axis("off")

for i, (step, w, x_center) in enumerate(zip(steps, box_widths, centers)):
    box = FancyBboxPatch(
        (x_center - w / 2, y_center - box_h / 2), w, box_h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.6, edgecolor="#1d3146", facecolor="#eaf1f8",
    )
    ax.add_patch(box)
    ax.text(x_center, y_center, step, ha="center", va="center",
             fontsize=fontsize, color="#13212e", fontweight="bold")

    if i < len(steps) - 1:
        next_x_center = centers[i + 1]
        next_w = box_widths[i + 1]
        arrow = FancyArrowPatch(
            (x_center + w / 2, y_center), (next_x_center - next_w / 2, y_center),
            arrowstyle="-|>", mutation_scale=18, linewidth=1.6, color="#2c3e50",
        )
        ax.add_patch(arrow)

fig.tight_layout()
fig.savefig("figures/pipeline_diagram.png", dpi=250, bbox_inches="tight", facecolor="white")
print("Saved to figures/pipeline_diagram.png")
