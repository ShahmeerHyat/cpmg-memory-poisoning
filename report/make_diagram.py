"""Draw the CPMG pipeline diagram -> report/figures/cpmg_diagram.png"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
BLUE, BLUE_L, ORANGE, ORANGE_L, GREY_L = "#2a78d6", "#e6f0fc", "#eb6834", "#fde9e0", "#efeeea"


def box(ax, x, y, w, h, title, body, edge, fill):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", linewidth=1.4,
                                edgecolor=edge, facecolor=fill))
    ax.text(x + w / 2, y + h - 0.13, title, ha="center", va="top", fontsize=8.5, color=INK, weight="bold")
    ax.text(x + w / 2, y + h - 0.40, body, ha="center", va="top", fontsize=6.8, color=INK2, linespacing=1.35)


def arrow(ax, a, b, color=INK2, text=None):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10, linewidth=1.2, color=color))
    if text:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.08, text, ha="center", fontsize=6.3, color=INK2)


def main():
    os.makedirs(OUT, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    box(ax, 0.1, 1.9, 1.7, 1.5, "Untrusted input", "email · slack · doc\nweb · calendar · tool\noutput", INK2, GREY_L)
    box(ax, 2.2, 1.9, 1.9, 1.5, "Provenance envelope", "system-recorded,\nnot editable by the LLM:\nchannel · authority tier\nage · parent tool call",
        INK2, GREY_L)
    box(ax, 4.5, 1.9, 2.1, 1.5, "Write checkpoint", "CPMG(provenance,\nmemory, source)\n-> risk score\n(store / quarantine)", BLUE, BLUE_L)
    box(ax, 7.0, 1.9, 1.3, 1.5, "Memory store", "records +\nenvelopes", INK2, GREY_L)
    box(ax, 4.5, 0.1, 2.1, 1.5, "Retrieval checkpoint", "CPMG(provenance, memory,\nsource, query / planned\naction, 3 co-retrieved\nmemories) -> risk score", BLUE, BLUE_L)
    box(ax, 7.0, 0.1, 2.85, 1.5, "Authority policy (deterministic)", "external / unknown memory can\nnever authorise a protected action:\npayment · send/share data\nchange settings · delete", ORANGE, ORANGE_L)
    box(ax, 0.1, 0.1, 3.9, 1.5, "Cross-encoder input", "[CLS] [CH_web][AUTH_external][AGE_*][PARENT_*]\nmemory [SEP] source interaction [SEP]\nquery / action [SEP] nbr1 [SEP] nbr2 [SEP] nbr3\nDeBERTa-v3-base cross-encoder", INK2, SURFACE)
    arrow(ax, (1.8, 2.65), (2.2, 2.65))
    arrow(ax, (4.1, 2.65), (4.5, 2.65))
    arrow(ax, (6.6, 2.65), (7.0, 2.65))
    ax.text(6.8, 2.78, "store", ha="center", fontsize=6.3, color=INK2)
    arrow(ax, (7.65, 1.9), (6.6, 1.2))
    ax.text(6.95, 1.68, "retrieve for\nquery / action", ha="right", fontsize=6.3, color=INK2)
    arrow(ax, (6.6, 0.85), (7.0, 0.85))
    fig.savefig(os.path.join(OUT, "cpmg_diagram.png"), dpi=220, bbox_inches="tight", facecolor=SURFACE)
    print("wrote", os.path.join(OUT, "cpmg_diagram.png"))


if __name__ == "__main__":
    main()
