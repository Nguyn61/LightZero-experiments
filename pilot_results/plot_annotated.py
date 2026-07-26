"""
Publication-quality learning curve plot
Actual data only + trend annotations (no fabricated predictions)
"""

import os, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

BASE1 = os.path.expanduser("~/lightzero-full/pilot_results")
BASE2 = os.path.expanduser("~/lightzero-full/pilot_results/sum_to_three_curriculum_reward")

ARMS   = ["A_sparse_fixed","B_dense_fixed","C_sparse_curriculum",
          "D_dense_curriculum","E_dense_broad"]
COLORS = ['#e53935','#1e88e5','#43a047','#fb8c00','#8e24aa']
LABELS = ["A — Sparse Fixed","B — Dense Fixed","C — Sparse Curriculum",
          "D — Dense Curriculum","E — Dense Broad"]

# Trend annotation config: (symbol, text, dy_offset, color_text)
TRENDS = {
    "A_sparse_fixed":      ("↗", "Still rising",   +0.12, '#e53935'),
    "B_dense_fixed":       ("→", "Plateaued",       -0.30, '#1e88e5'),
    "C_sparse_curriculum": ("↗", "Best — rising fast", +0.12, '#43a047'),
    "D_dense_curriculum":  ("↘", "Declining",        -0.35, '#fb8c00'),
    "E_dense_broad":       ("→", "Stabilizing",     +0.14, '#8e24aa'),
}

def load_arm(arm):
    for base in [BASE2, BASE1]:
        path = os.path.join(base, arm, "seed-0", "attempt-01", "log", "serial")
        if os.path.exists(path):
            files = glob.glob(os.path.join(path, "events.out.tfevents.*"))
            if files:
                ea = EventAccumulator(files[0]); ea.Reload()
                tag = 'evaluator_step/reward_mean'
                if tag in ea.Tags()['scalars']:
                    evs = ea.Scalars(tag)
                    return np.array([e.step for e in evs]), np.array([e.value for e in evs])
    return None, None

# ── Load all arms ────────────────────────────────────────────────────────────
data = {}
for arm in ARMS:
    steps, rewards = load_arm(arm)
    if steps is not None:
        data[arm] = (steps, rewards)

# ── Figure setup ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 7))
ax  = fig.add_subplot(111)

# Light grey "future zone" hint
pilot_end = max(s[-1] for s, _ in data.values())
ax.axvspan(pilot_end, pilot_end * 1.12, color='#f5f5f5', zorder=0)
ax.axvline(pilot_end, color='#bdbdbd', linewidth=1.0, linestyle='--', zorder=1)
ax.text(pilot_end + 200, 0.72, 'End of\nPilot run', fontsize=8,
        color='#9e9e9e', va='bottom')

# ── Plot each arm ────────────────────────────────────────────────────────────
for arm, color, label in zip(ARMS, COLORS, LABELS):
    if arm not in data:
        continue
    steps, rewards = data[arm]

    # Main line with markers
    ax.plot(steps, rewards,
            color=color, linewidth=2.5,
            marker='o', markersize=6, markerfacecolor='white',
            markeredgewidth=2, markeredgecolor=color,
            solid_capstyle='round', zorder=5, label=label)

    # Dashed trend extension (last 2 points slope)
    if len(steps) >= 2:
        dx = steps[-1] - steps[-2]
        dy = rewards[-1] - rewards[-2]
        slope = dy / dx
        # Clamp slope: declining → negative, plateau → ~0, rising → positive
        x_ext = np.array([steps[-1], steps[-1] + 2500])
        y_ext = rewards[-1] + slope * (x_ext - steps[-1])
        y_ext = np.clip(y_ext, 0, 15)
        ax.plot(x_ext, y_ext, color=color, linewidth=1.5,
                linestyle=':', alpha=0.6, zorder=4)

    # ── Trend annotation ──────────────────────────────────────────────────
    sym, txt, dy_off, ctxt = TRENDS[arm]
    xann = steps[-1] + 500
    yann = rewards[-1] + dy_off

    ax.annotate(
        f"  {sym} {txt}",
        xy=(steps[-1], rewards[-1]),
        xytext=(xann, yann),
        fontsize=9, color=ctxt, fontweight='bold',
        arrowprops=dict(arrowstyle='->', color=ctxt, lw=1.2),
        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=ctxt,
                  alpha=0.85, linewidth=0.8),
        zorder=8
    )

# ── Final reward labels on right axis ────────────────────────────────────────
for arm, color in zip(ARMS, COLORS):
    if arm not in data:
        continue
    steps, rewards = data[arm]
    ax.text(steps[-1] - 300, rewards[-1] + 0.06,
            f"{rewards[-1]:.2f}", fontsize=8.5, color=color,
            fontweight='bold', ha='right', va='bottom', zorder=9)

# ── Decorations ───────────────────────────────────────────────────────────────
ax.set_xlabel("Environment Steps", fontsize=12, labelpad=8)
ax.set_ylabel("Eval Reward Mean", fontsize=12, labelpad=8)
ax.set_title(
    "Sum-to-Three Curriculum Reward — Pilot Learning Curves\n"
    r"$\it{Seed\ 0,\ 15k\ steps\ each\ arm}$",
    fontsize=13, fontweight='bold', pad=14)

ax.legend(fontsize=9.5, loc='upper left',
          framealpha=0.9, edgecolor='#bdbdbd')
ax.grid(True, alpha=0.2, linestyle='--')
ax.set_xlim(left=-300)
ax.set_ylim(bottom=0.5)

# Custom x-axis: show step counts nicely
import matplotlib.ticker as mticker
ax.xaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))))

# Footnote
fig.text(0.5, -0.02,
         "Note: dotted extensions show local slope trend — not a fitted prediction. "
         "Main runs (P5) required for statistical inference.",
         ha='center', fontsize=8, color='#757575', style='italic')

plt.tight_layout()
out = os.path.join(BASE1, "learning_curves_annotated.png")
plt.savefig(out, dpi=160, bbox_inches='tight')
print(f"✅ Saved → {out}")

# Print summary table
print("\n── Final reward summary ──")
print(f"{'Arm':<28} {'Final Reward':>12}  Trend")
print("─" * 55)
for arm, label in zip(ARMS, LABELS):
    if arm not in data:
        continue
    steps, rewards = data[arm]
    sym, txt, *_ = TRENDS[arm]
    print(f"{label:<28} {rewards[-1]:>12.2f}  {sym} {txt}")
