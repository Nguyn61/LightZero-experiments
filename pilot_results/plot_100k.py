import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from scipy.optimize import curve_fit

base_dir1 = os.path.expanduser("~/lightzero-full/pilot_results")
base_dir2 = os.path.expanduser("~/lightzero-full/pilot_results/sum_to_three_curriculum_reward")
arms = ["A_sparse_fixed", "B_dense_fixed", "C_sparse_curriculum", "D_dense_curriculum", "E_dense_broad"]
colors = ['red', 'blue', 'green', 'orange', 'purple']

# Hàm học dạng logarithm: y = a * log(x + 1) + b (phổ biến nhất trong RL)
def log_curve(x, a, b):
    return a * np.log(x + 1) + b

# Hàm học dạng sqrt: y = a * sqrt(x) + b
def sqrt_curve(x, a, b):
    return a * np.sqrt(x) + b

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# --- Plot 1: Đường thực tế + Dự đoán ---
ax1 = axes[0]
ax1.set_title("Thực tế (nét liền) + Dự đoán đến 100k steps (nét đứt)", fontsize=12)
ax1.set_xlabel("Environment Steps")
ax1.set_ylabel("Eval Reward Mean")
ax1.grid(True, alpha=0.3)

future_steps = np.linspace(0, 100000, 500)
predictions = {}

for arm, color in zip(arms, colors):
    events_dir = os.path.join(base_dir2, arm, "seed-0", "attempt-01", "log", "serial")
    if not os.path.exists(events_dir):
        events_dir = os.path.join(base_dir1, arm, "seed-0", "attempt-01", "log", "serial")
    if not os.path.exists(events_dir):
        continue

    event_files = glob.glob(os.path.join(events_dir, "events.out.tfevents.*"))
    if not event_files:
        continue

    ea = EventAccumulator(event_files[0])
    ea.Reload()
    tag = 'evaluator_step/reward_mean'
    if tag not in ea.Tags()['scalars']:
        continue

    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    rewards = np.array([e.value for e in events])

    # Vẽ đường thực tế
    ax1.plot(steps, rewards, color=color, linewidth=2.5, label=f"{arm}")

    # Fit log curve
    try:
        popt, _ = curve_fit(log_curve, steps, rewards, p0=[1, 0], maxfev=5000)
        y_pred = log_curve(future_steps, *popt)
        # Cap ở mức hợp lý (max ~15 vì game có 10 bước)
        y_pred = np.clip(y_pred, 0, 15)
        ax1.plot(future_steps, y_pred, color=color, linewidth=1.5, linestyle='--', alpha=0.7)
        
        pred_100k = float(log_curve(100000, *popt))
        pred_100k = max(0, min(15, pred_100k))
        predictions[arm] = pred_100k
        print(f"{arm}: Dự đoán tại 100k steps = {pred_100k:.2f}")
    except Exception as e:
        print(f"Fit lỗi cho {arm}: {e}")

ax1.axvline(x=10000, color='gray', linestyle=':', linewidth=1.5, label='← Pilot | Dự đoán →')
ax1.legend(fontsize=9)

# --- Plot 2: Cột so sánh tại 100k steps ---
ax2 = axes[1]
ax2.set_title("Dự đoán Reward tại 100,000 steps", fontsize=12)
arm_labels = [a.replace('_', '\n') for a in predictions.keys()]
values = list(predictions.values())
bar_colors = [colors[arms.index(a)] for a in predictions.keys()]

bars = ax2.bar(arm_labels, values, color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.8)

# Ghi số lên đầu cột
for bar, val in zip(bars, values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

ax2.set_ylabel("Predicted Reward Mean")
ax2.set_ylim(0, max(values) * 1.2)
ax2.grid(True, alpha=0.3, axis='y')

plt.suptitle("Sum-to-Three Curriculum Reward — Scaling Analysis (Pilot → 100k steps)", fontsize=13, fontweight='bold')
plt.tight_layout()
out_path = os.path.join(base_dir1, "learning_curves_100k.png")
plt.savefig(out_path, dpi=150)
print(f"\nPlot saved to {out_path}")
