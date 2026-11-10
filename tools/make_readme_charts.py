"""The README's benchmark chart (docs/img/readme_benchmark.png). Numbers are the outcome counts tools/bench_notes.py
gave on 2026-11-10 (three systems on the free-form note sets, tuner damage guard on); edit them here when a new run
replaces them.

    python tools/make_readme_charts.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
# (all checks met, nothing else moved) / (all checks met) / (damage flag) / (a check missed)
SETS = [("Sets A + B: 48 blind notes", [
            ("Rules, no model", (0, 0, 0, 48)), ("Qwen2.5-7B, local", (2, 30, 1, 15)), ("A large model", (6, 39, 2, 1))]),
        ("Set C: 40 blind notes, written by a model", [
            ("Rules, no model", (5, 0, 0, 35)), ("Qwen2.5-7B, local", (1, 32, 0, 7)), ("A large model", (11, 25, 1, 3))])]
COLS = ("#2e7d32", "#81c784", "#e0a030", "#c0504d")
KEYS = ("right way, nothing else moved", "right way, no damage", "damaged (held for a person)", "missed a check")


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.4), dpi=160, sharey=True)
    for ax, (title, rows) in zip(axes, SETS):
        for i, (name, counts) in enumerate(rows):
            left, n = 0, sum(counts)
            for c, col, key in zip(counts, COLS, KEYS):
                ax.barh(i, c, left=left, color=col, label=key if i == 0 else None)
                if c >= 3:
                    ax.text(left + c / 2, i, str(c), ha="center", va="center", fontsize=9,
                            color="black" if col == COLS[1] else "white")
                left += c
            ax.text(n + 0.6, i, f"{(counts[0] + counts[1]) / n:.0%}", va="center", fontsize=10, weight="bold")
        ax.set_yticks(range(len(rows)), [r[0] for r in rows])
        ax.invert_yaxis()
        ax.set_xlim(0, sum(rows[0][1]) * 1.12)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("notes")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Did the grade move the picture the way the note asked, without damage?", fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    out = ROOT / "docs" / "img" / "readme_benchmark.png"
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()
