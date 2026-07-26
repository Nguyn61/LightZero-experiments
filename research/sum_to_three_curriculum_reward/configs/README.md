# Cấu hình nghiên cứu

## Trạng thái

- **Implemented:** [`experiment_matrix.json`](experiment_matrix.json) là protocol machine-readable của study.
- **Implemented:** runnable config factory nằm tại [`../../../zoo/pooltool/sum_to_three/config/research_alpha.py`](../../../zoo/pooltool/sum_to_three/config/research_alpha.py).
- **Not yet verified by a full training run:** môi trường hiện tại của phiên làm việc không có `pooltool`/`pytest`; cần chạy smoke test trong WSL/Vast.ai environment đã cài dependencies.

## Experiment arms

| Arm | Collector reward | Collector start distribution |
| --- | --- | --- |
| `A_sparse_fixed` | `binary` | `canonical` |
| `B_dense_fixed` | `event_aligned` | `canonical` |
| `C_sparse_curriculum` | `binary` | `canonical -> local -> broad` |
| `D_dense_curriculum` | `event_aligned` | `canonical -> local -> broad` |
| `E_dense_broad` | `event_aligned` | `broad` ngay từ đầu |

Built-in evaluator luôn bị ép về `binary + canonical`. Script external evaluation có thể yêu cầu `local` hoặc `broad`, nhưng reward vẫn là `binary`.

## Commands từ repository root

Smoke CPU, 2,000 environment steps:

```bash
python zoo/pooltool/sum_to_three/config/research_alpha.py \
  --arm A_sparse_fixed --seed 0 --max-env-step 2000 --cpu
```

Full-method smoke:

```bash
python zoo/pooltool/sum_to_three/config/research_alpha.py \
  --arm D_dense_curriculum --seed 0 --max-env-step 2000 --cpu
```

Pilot trên GPU:

```bash
python zoo/pooltool/sum_to_three/config/research_alpha.py \
  --arm D_dense_curriculum --seed 0 --max-env-step 15000
```

Main run đề xuất:

```bash
python zoo/pooltool/sum_to_three/config/research_alpha.py \
  --arm D_dense_curriculum --seed 0 --max-env-step 40000
```

Không thay K, MCTS simulations, replay ratio, reanalyze ratio, SSL weight hoặc learning rate giữa các arm.

## Diagnostics và evaluation

```bash
python research/sum_to_three_curriculum_reward/scripts/random_policy_diagnostics.py \
  --episodes 100 \
  --output research/sum_to_three_curriculum_reward/records/random-policy-seed0.json
```

```bash
python research/sum_to_three_curriculum_reward/scripts/evaluate_checkpoint.py \
  experiments/sum_to_three_curriculum_reward/D_dense_curriculum/seed-0/attempt-01/ckpt/ckpt_best.pth.tar \
  research/sum_to_three_curriculum_reward/records/D-seed0-T0.json \
  --run-id D_dense_curriculum-seed-0-attempt-01 \
  --arm D_dense_curriculum --seed 0 --episodes 100 --test-distribution canonical
```

Protocol chi tiết: [experiment_matrix.json](experiment_matrix.json). Mỗi run phải có manifest theo [`../templates/run_manifest.json`](../templates/run_manifest.json).
