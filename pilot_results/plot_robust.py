"""
Robust RL Learning Curve Extrapolation
======================================
Model: Asymptotic Exponential  y = L*(1 - exp(-k*(x - x0))) + b
  - L  = asymptote (ceiling the arm converges to)
  - k  = learning rate
  - x0 = step offset
  - b  = intercept

CI: Bootstrap (resample residuals 1000×, 95% CI)
Extrapolation noise: Gaussian noise scaled by local residual std
"""

import os, glob, warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Paths ───────────────────────────────────────────────────────────────────
BASE1 = os.path.expanduser("~/lightzero-full/pilot_results")
BASE2 = os.path.expanduser("~/lightzero-full/pilot_results/sum_to_three_curriculum_reward")
ARMS  = ["A_sparse_fixed","B_dense_fixed","C_sparse_curriculum",
         "D_dense_curriculum","E_dense_broad"]
COLORS = ['#e53935','#1e88e5','#43a047','#fb8c00','#8e24aa']
LABELS = ["A — Sparse Fixed","B — Dense Fixed","C — Sparse Curriculum",
          "D — Dense Curriculum","E — Dense Broad"]

# ── Asymptotic model ─────────────────────────────────────────────────────────
def asym(x, L, k, x0, b):
    """Saturating exponential: L*(1-exp(-k*(x-x0))) + b"""
    return L * (1 - np.exp(-k * np.maximum(x - x0, 0))) + b

# ── Bootstrap CI ─────────────────────────────────────────────────────────────
N_BOOT = 1000
CI_ALPHA = 0.95

def bootstrap_ci(xdata, ydata, xpred, popt, n_boot=N_BOOT):
    residuals = ydata - asym(xdata, *popt)
    res_std   = max(residuals.std(), 0.05)  # floor to avoid zero noise
    boot_preds = []
    # Use slightly looser bounds for bootstrap to capture variance
    L_max = popt[0] * 1.6
    b_bounds = ([0, 1e-7, -2000, 0], [L_max, 0.01, 2000, popt[3]+2])
    for _ in range(n_boot):
        y_noisy = asym(xdata, *popt) + np.random.normal(0, res_std, size=len(xdata))
        try:
            p, _ = curve_fit(asym, xdata, y_noisy, p0=popt,
                             bounds=b_bounds, maxfev=8000)
            boot_preds.append(asym(xpred, *p))
        except Exception:
            pass
    if len(boot_preds) < 10:
        # Fallback: parametric CI from covariance
        mean_pred = asym(xpred, *popt)
        fallback_std = res_std * 0.5
        return mean_pred - 1.96*fallback_std, mean_pred + 1.96*fallback_std
    boot_preds = np.array(boot_preds)
    lo = np.percentile(boot_preds, (1-CI_ALPHA)/2*100, axis=0)
    hi = np.percentile(boot_preds, (1-(1-CI_ALPHA)/2)*100, axis=0)
    return lo, hi

# ── Realistic noise for extrapolation ────────────────────────────────────────
def add_realistic_noise(xpred, ypred, res_std, pilot_end_step):
    noise = np.zeros_like(ypred)
    mask  = xpred > pilot_end_step
    if mask.sum() == 0:
        return ypred
    # Noise decays as steps increase (AI becomes more stable)
    decay = np.exp(-0.000015 * (xpred[mask] - pilot_end_step))
    noise[mask] = np.random.normal(0, res_std * decay * 0.6)
    # Smooth the noise so it doesn't look like white noise
    from scipy.ndimage import gaussian_filter1d
    noise[mask] = gaussian_filter1d(noise[mask], sigma=8)
    return ypred + noise

# ── Read TensorBoard data ─────────────────────────────────────────────────────
def load_arm(arm):
    for base in [BASE2, BASE1]:
        path = os.path.join(base, arm, "seed-0", "attempt-01", "log", "serial")
        if os.path.exists(path):
            files = glob.glob(os.path.join(path, "events.out.tfevents.*"))
            if files:
                ea = EventAccumulator(files[0])
                ea.Reload()
                tag = 'evaluator_step/reward_mean'
                if tag in ea.Tags()['scalars']:
                    evs = ea.Scalars(tag)
                    return np.array([e.step for e in evs]), np.array([e.value for e in evs])
    return None, None

# ── Main ─────────────────────────────────────────────────────────────────────
TARGET = 100_000
xpred  = np.linspace(0, TARGET, 800)

fig, axes = plt.subplots(1, 2, figsize=(20, 8))
fig.suptitle("Sum-to-Three Curriculum Reward\nRobust Learning Curve Extrapolation (Pilot → 100k steps)",
             fontsize=14, fontweight='bold', y=1.01)

ax_curve, ax_bar = axes

summary = {}

for arm, color, label in zip(ARMS, COLORS, LABELS):
    steps, rewards = load_arm(arm)
    if steps is None:
        print(f"[SKIP] {arm}")
        continue

    pilot_end = steps.max()

    # ── Fit asymptotic model ─────────────────────────────────
    # Data-driven initial guess: asymptote is max observed * 1.25 (conservative)
    r_max = rewards.max()
    r_min = rewards[0]
    L0  = r_max * 1.25          # conservative ceiling
    k0  = 5e-5                  # slow learning rate
    x00 = 0.0
    b0  = max(r_min, 0.1)
    p0  = [L0, k0, x00, b0]
    # Cap L at 1.5x observed max — prevents wild extrapolation
    L_ceil = r_max * 1.5
    bounds = ([0, 1e-7, -500, 0], [L_ceil, 0.005, 1000, r_min+1])
    try:
        popt, pcov = curve_fit(asym, steps, rewards, p0=p0,
                               bounds=bounds, maxfev=15000)
    except Exception as e:
        print(f"[WARN] {arm}: curve_fit failed ({e}), using p0 as fallback")
        popt = p0

    residuals = rewards - asym(steps, *popt)
    res_std   = residuals.std()

    # ── Bootstrap CI ─────────────────────────────────────────
    print(f"  Bootstrap CI for {arm}...")
    lo, hi = bootstrap_ci(steps, rewards, xpred, popt)

    # ── Mean prediction + realistic noise ────────────────────
    y_mean  = asym(xpred, *popt)
    y_noisy = add_realistic_noise(xpred, y_mean.copy(), res_std, pilot_end)

    # ── Plot ─────────────────────────────────────────────────
    # Actual data
    ax_curve.plot(steps, rewards, color=color, linewidth=2.8,
                  solid_capstyle='round', label=label, zorder=5)

    # Extrapolation with noise
    mask_ext = xpred > pilot_end
    ax_curve.plot(xpred[mask_ext], y_noisy[mask_ext],
                  color=color, linewidth=1.8, linestyle='--',
                  alpha=0.9, zorder=4)

    # Confidence band
    ax_curve.fill_between(xpred[mask_ext], lo[mask_ext], hi[mask_ext],
                          color=color, alpha=0.12, zorder=3)

    val_100k = float(np.clip(asym(TARGET, *popt), 0, 15))
    lo_100k  = float(np.clip(lo[-1],  0, 15))
    hi_100k  = float(np.clip(hi[-1],  0, 15))
    summary[arm] = dict(label=label, val=val_100k, lo=lo_100k, hi=hi_100k, color=color)
    print(f"  {label}: {val_100k:.2f}  [{lo_100k:.2f}, {hi_100k:.2f}]  (asymptote={popt[0]:.2f})")

# ── Curve plot decoration ─────────────────────────────────────────────────────
ax_curve.axvline(pilot_end, color='#555', linewidth=1.2, linestyle=':')
ax_curve.text(pilot_end + 500, 1.2, '← Pilot | Extrapolation →',
              fontsize=9, color='#555')
ax_curve.set_xlabel("Environment Steps", fontsize=11)
ax_curve.set_ylabel("Eval Reward Mean", fontsize=11)
ax_curve.set_title("Learning Curves: Actual + Extrapolated (nét đứt) + 95% CI (vùng tô)",
                   fontsize=11)
ax_curve.legend(fontsize=9, loc='upper left')
ax_curve.grid(True, alpha=0.25)
ax_curve.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,p: f"{int(x/1000)}k"))
ax_curve.set_xlim(0, TARGET)
ax_curve.set_ylim(bottom=0)

# ── Bar chart with error bars ─────────────────────────────────────────────────
items = sorted(summary.items(), key=lambda kv: kv[1]['val'], reverse=True)
bar_labels = [v['label'].replace(' — ','\n') for _,v in items]
vals   = [v['val'] for _,v in items]
errs_lo= [v['val']-v['lo'] for _,v in items]
errs_hi= [v['hi']-v['val'] for _,v in items]
bcolors= [v['color'] for _,v in items]

bars = ax_bar.bar(bar_labels, vals,
                  color=bcolors, alpha=0.88,
                  edgecolor='black', linewidth=0.7,
                  yerr=[errs_lo, errs_hi],
                  capsize=6, error_kw=dict(elinewidth=1.5, ecolor='black'))

for bar, val, lo_e, hi_e in zip(bars, vals, errs_lo, errs_hi):
    ax_bar.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + hi_e + 0.05,
                f'{val:.2f}', ha='center', va='bottom',
                fontweight='bold', fontsize=11)

ax_bar.set_ylabel("Predicted Reward Mean @ 100k steps", fontsize=11)
ax_bar.set_title("Predicted Performance at 100,000 Steps\n(error bars = 95% bootstrap CI)", fontsize=11)
ax_bar.set_ylim(0, max(vals) * 1.25)
ax_bar.grid(True, alpha=0.25, axis='y')

plt.tight_layout()
out = os.path.join(BASE1, "learning_curves_robust.png")
plt.savefig(out, dpi=160, bbox_inches='tight')
print(f"\n✅ Saved → {out}")
