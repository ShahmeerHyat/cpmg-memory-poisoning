"""Draw the data figures of the paper from results/metrics.json -> paper/figures/*.pdf (vector).

  capability_matrix.pdf   who can do what: models x tasks, test AUROC (sequential single hue; value printed in each cell)
  generalisation.pdf      activation AUROC as the test moves away from the training wording (3 series, direct labels)
Palette: reference categorical slots 1-3 (blue, orange, aqua) in fixed order; sequential = one blue ramp.
Usage: python paper/make_figures.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
AGG = json.load(open(os.path.join(ROOT, "results/metrics.json")))["over_seeds"]
OUT = os.path.join(HERE, "figures")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#ffffff", "#cfe1f7", "#7fb0ea", "#2a78d6", "#1a56a6"])
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
                     "pdf.fonttype": 42, "font.size": 8, "axes.linewidth": 0.6})


def get(m, s, k="auroc"):
    a = AGG.get(m, {}).get(s, {}).get(k)
    return None if a is None else a["mean"]


def capability_matrix():
    rows = [("CPMG (retrieval)", "cpmg_full", 0), ("CPMG write+retrieval", "cpmg_write_retrieval", 0),
            ("$-$ provenance", "abl_no_prov", 1), ("$-$ query", "abl_no_query", 1), ("$-$ neighbours", "abl_no_nbrs", 1),
            ("write-time only", "write_only", 1),
            ("Content-only DeBERTa", "content_only", 2), ("PIGuard (zero-shot)", "piguard", 2), ("Keyword filter", "keyword", 2),
            ("TF-IDF (memory)", "tfidf_memory", 2), ("TF-IDF (full input)", "tfidf_full", 2),
            ("Copy rule", "lexmatch", 3), ("Quoted-copy rule$^{\\dagger}$", "lexquote", 3)]
    cols = [("L1\ndirect", "P_L1"), ("L2\npayload in\na neighbour", "P_L2"),
            ("L3\ntrigger in\nthe query", "A_L3_hard_real"), ("L2+L3\nboth", "A_L23_hard_real")]
    groups = ["CPMG", "Ablations", "Record-only and\nlexical baselines", "Hand-written rules\n(activation only)"]
    fig, ax = plt.subplots(figsize=(3.35, 4.45))
    y = 0.0
    ys, prev = [], None
    for name, m, g in rows:
        if prev is not None and g != prev:
            y += 0.28
        ys.append(y)
        prev = g
        y += 1
    for (name, m, g), yy in zip(rows, ys):
        for j, (_, s) in enumerate(cols):
            v = get(m, s)
            if g == 3 and j < 2:
                v = None  # the rules have no poison score
            if v is None:
                ax.add_patch(plt.Rectangle((j + 0.03, yy + 0.04), 0.94, 0.92, facecolor="#f3f2ee", edgecolor="none"))
                ax.text(j + 0.5, yy + 0.5, "n/a", ha="center", va="center", fontsize=7, color=INK2)
                continue
            t = min(1.0, max(0.0, (v - 0.5) / 0.5))
            ax.add_patch(plt.Rectangle((j + 0.03, yy + 0.04), 0.94, 0.92, facecolor=SEQ(t), edgecolor="none"))
            ax.text(j + 0.5, yy + 0.5, f"{v:.3f}", ha="center", va="center", fontsize=7.5,
                    color="white" if t > 0.6 else INK, fontweight="bold" if t > 0.95 else "normal")
        ax.text(-0.08, yy + 0.5, name, ha="right", va="center", fontsize=7.5, color=INK)
    for j, (lab, _) in enumerate(cols):
        ax.text(j + 0.5, -0.15, lab, ha="center", va="bottom", fontsize=7, color=INK, linespacing=1.05)
    ax.text(1.0, -1.75, "Poison detection", ha="center", va="bottom", fontsize=7.5, color=INK2, fontweight="bold")
    ax.text(3.0, -1.75, "Trigger activation\n(unseen real triggers)", ha="center", va="bottom", fontsize=7.5, color=INK2,
            fontweight="bold", linespacing=1.05)
    ax.plot([0.05, 1.95], [-1.62, -1.62], color=INK2, lw=0.6, clip_on=False)
    ax.plot([2.05, 3.95], [-1.62, -1.62], color=INK2, lw=0.6, clip_on=False)
    ax.set_xlim(0, 4)
    ax.set_ylim(y, -0.05)
    ax.axis("off")
    # colour key
    cax = fig.add_axes([0.42, 0.035, 0.42, 0.018])
    cax.imshow(np.linspace(0, 1, 100)[None, :], aspect="auto", cmap=SEQ, extent=[0.5, 1.0, 0, 1])
    cax.set_yticks([])
    cax.set_xticks([0.5, 0.75, 1.0], ["0.5 (chance)", "0.75", "1.0"], fontsize=6.5, color=INK2)
    cax.tick_params(length=2, width=0.5, color=INK2)
    for sp in cax.spines.values():
        sp.set_linewidth(0.4)
        sp.set_color(INK2)
    fig.text(0.40, 0.044, "AUROC", ha="right", va="center", fontsize=7, color=INK2)
    fig.subplots_adjust(left=0.36, right=0.995, top=0.835, bottom=0.085)
    fig.savefig(os.path.join(OUT, "capability_matrix.pdf"))
    plt.close(fig)


def generalisation():
    conds = [("Random\ntriggers", "A_L3_hard_rand", "cpmg_full"), ("Real\ntriggers", "A_L3_hard_real", "cpmg_full"),
             ("Reworded\nattack", "A_L3_hard@para", "cpmg_full"), ("Reworded,\nno quotes", "A_L3_hard@paranq", "cpmg_full")]
    series = [("CPMG", None, BLUE), ("Quoted-copy rule$^{\\dagger}$", "lexquote", ORANGE), ("Copy rule", "lexmatch", AQUA)]
    if get("cpmg_full", "A_L3_hard@para") is None:
        print("generalisation.pdf skipped: paraphrase predictions missing")
        return
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    w = 0.26
    for k, (name, m, col) in enumerate(series):
        vals = [get(m or cm, s) for _, s, cm in conds]
        x = np.arange(len(conds)) + (k - 1) * w
        ax.bar(x, [v - 0.4 for v in vals], bottom=0.4, width=w - 0.035, color=col, label=name, linewidth=0)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.008, f"{v:.2f}", ha="center", va="bottom", fontsize=6.3, color=INK)
    ax.axhline(0.5, color=INK2, lw=0.7, ls=(0, (4, 2)))
    ax.set_xticks(range(len(conds)), [c[0] for c in conds], fontsize=6.8, color=INK)
    ax.set_ylim(0.4, 1.1)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_ylabel("Activation AUROC (test)", fontsize=7.2, color=INK2)
    ax.tick_params(axis="y", labelsize=6.8, colors=INK2, length=2, width=0.5)
    ax.tick_params(axis="x", length=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
    ax.yaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.axvline(1.5, color=GRID, lw=0.8)
    ax.text(0.5, 1.085, "original wording, unseen triggers", ha="center", va="top", fontsize=6.3, color=INK2)
    ax.text(2.5, 1.085, "attack reworded by an LLM", ha="center", va="top", fontsize=6.3, color=INK2)
    ax.legend(fontsize=6.6, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01), handlelength=1.0,
              columnspacing=1.0, handletextpad=0.4)
    fig.subplots_adjust(left=0.135, right=0.995, top=0.9, bottom=0.17)
    fig.savefig(os.path.join(OUT, "generalisation.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    capability_matrix()
    generalisation()
    print("figures written to", OUT)
