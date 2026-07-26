---
schema_version: 1
project_id: pooltool-sum-to-three-curriculum-reward
status: implemented-not-runtime-verified
package_root: research/sum_to_three_curriculum_reward
baseline_commit: 272acb9d73b10d67baaa31a0a878408a4147974e
primary_metric: binary_episode_score_mean
primary_baseline: A_sparse_fixed
protocol: configs/experiment_matrix.json
---

# Agent boot context — đọc file này trước

Mục tiêu của file này là giúp một future agent bắt đầu từ số 0 mà không nhầm proposal với kết quả đã chạy. Không suy diễn trạng thái ngoài các nhãn dưới đây.

## Status vocabulary

- **Implemented:** code/file đang có trong working tree.
- **Verified observation:** sự thật đã đọc trực tiếp từ code hoặc artifact.
- **Runtime verified:** đã chạy thành công trong environment có dependencies.
- **Planned:** chưa triển khai hoặc chưa có kết quả.

## Current status

### Implemented trong working tree

- `event_aligned` reward và reusable `ShotOutcome` tại [`../../zoo/pooltool/sum_to_three/envs/utils.py`](../../zoo/pooltool/sum_to_three/envs/utils.py).
- Start distributions `canonical`, `local`, `broad`, deterministic environment-owned RNG và fixed curriculum tại [`../../zoo/pooltool/sum_to_three/envs/sum_to_three_env.py`](../../zoo/pooltool/sum_to_three/envs/sum_to_three_env.py).
- Collector/evaluator role separation: built-in evaluator luôn `binary + canonical`; external evaluator có thể dùng `local`/`broad` nhưng reward vẫn binary.
- Episode diagnostics: binary return, sparse successes, contacts và cushion histogram.
- Unit-test cases mới tại [`../../zoo/pooltool/sum_to_three/envs/test_sum_to_three_env.py`](../../zoo/pooltool/sum_to_three/envs/test_sum_to_three_env.py).
- A–E config factory tại [`../../zoo/pooltool/sum_to_three/config/research_alpha.py`](../../zoo/pooltool/sum_to_three/config/research_alpha.py).
- Student kit, standalone tracker, notebook, templates và scripts trong folder này.

### Verified observation

- Baseline task dùng 4-D coordinate observation `[cue_x, cue_y, object_x, object_y]`.
- Policy action là 2-D normalized `[-1,1]`; environment scale sang cue speed `[0.3,3.0]` và cut angle `[-70,70]`.
- Spin/elevation vẫn bằng 0.
- Episode dài 10 shots.
- Binary success giữ nguyên: có cue-object `BALL_BALL` event và tổng cue+object `BALL_LINEAR_CUSHION` events đúng bằng 3.
- Simulator giữ safeguard `max_events=200`.
- Frozen SEZ baseline: K=20, 50 simulations, replay=1, reanalyze=.25, SSL=2, LR=.003.
- Không có quantitative training result/checkpoint được commit tại thời điểm khảo sát.

### Chưa runtime verified trong session Windows hiện tại

- Targeted `pytest` không chạy vì Python hiện tại thiếu `pytest`.
- Import smoke không chạy vì Python hiện tại thiếu `pooltool`.
- `py_compile` cho các Python file mới/sửa đã pass.
- Do đó không được nói curriculum/reward đã train thành công cho tới khi chạy WSL/Vast.ai smoke protocol.

### Planned

- Chạy baseline audit và A/D 2k smoke trong environment đầy đủ.
- Chạy random-policy diagnostics.
- Pilot/main multi-seed runs.
- Literature review primary-source; workflow web trước đó không trích xuất được claim đáng tin cậy.
- Paper/report dựa trên results thật.

## Scientific thesis và scope

> Event-aligned reward densification cộng progressive start-state curriculum có thể cải thiện binary sample efficiency và held-out start-state generalization của Sampled EfficientZero trên SumToThree.

Không làm trong study này:

- sửa Sampled EfficientZero/MCTS/replay buffer;
- spin, elevation hoặc image observation như contribution;
- SAC tuning rộng;
- claim thuật toán mới, “first” hoặc SOTA.

## Source map

| Concern | Source of truth |
| --- | --- |
| Reward/outcome | `zoo/pooltool/sum_to_three/envs/utils.py`: `ShotOutcome`, `get_shot_outcome`, `binary`, `event_aligned` |
| Start sampler/curriculum | `zoo/pooltool/sum_to_three/envs/sum_to_three_env.py`: `StartDistribution`, `sample_initial_positions`, `SumToThreeEnv._resolve_start_distribution` |
| Collector/evaluator split | `SumToThreeEnv.create_collector_env_cfg`, `create_evaluator_env_cfg` |
| Episode diagnostics | `EpisodicTrackedStats`, `SumToThreeEnv.step` |
| Tests | `zoo/pooltool/sum_to_three/envs/test_sum_to_three_env.py` |
| Research arms | `zoo/pooltool/sum_to_three/config/research_alpha.py` |
| Frozen protocol | `configs/experiment_matrix.json` |
| Run manifest/result schema | `templates/*.json` |
| Human progress | `progress_tracker.html` — không phải scientific source of truth |
| Stack rationale | `WHY_LIGHTZERO_POOLTOOL.md` — vì sao LightZero/PoolTool/SumToThree |
| Paper positioning | `PAPER_GUIDE.md` — contribution tiers, sections, claim discipline |
| Future directions | `FUTURE_THESES.md` — thesis catalog theo tier rủi ro |

## Reward contract

| Outcome | `binary` | `event_aligned` |
| --- | ---: | ---: |
| no object contact | 0.0 | 0.0 |
| contact + 0 cushion | 0.0 | 0.1 |
| contact + 1 cushion | 0.0 | 0.2 |
| contact + 2 cushions | 0.0 | 0.3 |
| contact + exactly 3 | 1.0 | 1.0 |
| contact + >3 | 0.0 | 0.1 |

`event_aligned` là reward densification, không phải policy-invariant shaping. Headline evaluation luôn `binary`.

## Curriculum contract

- `canonical`: layout gốc.
- `local`: perturb quanh canonical theo 8% usable table dimensions.
- `broad`: uniform valid positions trong table.
- Mọi randomized sample phải thỏa ball separation `>= 2R + margin`.
- Fixed collector schedule: 25% canonical, 40% local, 35% broad.
- Built-in evaluator không advance curriculum.
- RNG thuộc environment và được reset bởi `env.seed(...)`; không dựa vào global `np.random` cho start-state sequence.

## Experiment matrix

| Arm | Collector reward | Collector starts |
| --- | --- | --- |
| A | binary | canonical |
| B | event_aligned | canonical |
| C | binary | curriculum |
| D | event_aligned | curriculum |
| E optional | event_aligned | broad from start |

Primary comparison: A vs D. A–D là factorial ablation. E kiểm tra curriculum ordering có hơn random diversity hay không.

## Commands từ repo root

### Environment tests

```bash
python -m pytest zoo/pooltool/sum_to_three/envs/test_sum_to_three_env.py -q
```

### Random-policy diagnostic

```bash
python research/sum_to_three_curriculum_reward/scripts/random_policy_diagnostics.py \
  --episodes 100 \
  --output research/sum_to_three_curriculum_reward/records/random-policy-seed0.json
```

### Baseline/full-method smoke

```bash
python zoo/pooltool/sum_to_three/config/research_alpha.py \
  --arm A_sparse_fixed --seed 0 --max-env-step 2000

python zoo/pooltool/sum_to_three/config/research_alpha.py \
  --arm D_dense_curriculum --seed 0 --max-env-step 2000
```

Thêm `--cpu` nếu CUDA chưa sẵn sàng.

### TensorBoard export

```bash
python research/sum_to_three_curriculum_reward/scripts/export_tensorboard.py \
  experiments/sum_to_three_curriculum_reward/A_sparse_fixed/seed-0/attempt-01/log/serial \
  research/sum_to_three_curriculum_reward/records/A-seed0-scalars.json \
  --run-id A-seed0 --arm A_sparse_fixed --seed 0
```

### Checkpoint evaluation

```bash
python research/sum_to_three_curriculum_reward/scripts/evaluate_checkpoint.py \
  <checkpoint.pth.tar> \
  research/sum_to_three_curriculum_reward/records/D-seed0-T0.json \
  --run-id D_dense_curriculum-seed-0-attempt-01 \
  --arm D_dense_curriculum --seed 0 --episodes 100 \
  --test-distribution canonical
```

### Student tools

```bash
jupyter notebook research/sum_to_three_curriculum_reward/notebooks/experiment_lab.ipynb
```

Mở `progress_tracker.html` trực tiếp bằng browser.

## Guardrails

1. Không thay primary metric sau khi xem kết quả.
2. Không dùng shaped collector return làm headline result.
3. Không chọn checkpoint bằng held-out T0/T1/T2 final test.
4. Không extend budget riêng cho arm ưa thích.
5. Không overwrite raw logs/checkpoints/run records.
6. Không sửa MCTS/policy/replay buffer cho thesis này.
7. Không gọi feature “verified” nếu mới chỉ compile.
8. Không claim novelty trước literature review.
9. Ghi mọi failed/excluded run và lý do.

## Next-task queue

1. **Dependency owner:** activate WSL/Vast.ai Python 3.10 env; verify `pooltool`, LightZero, pytest, CUDA.
2. **Test owner:** run targeted environment tests; fix failures without changing binary semantics.
3. **Smoke owner:** run A and D for 2k steps; verify evaluator is canonical+binary and curriculum reaches configured stages.
4. **Diagnostics owner:** run 100-episode random diagnostics for canonical/local/broad; inspect event rarity.
5. **Artifact owner:** create manifest + journal records for every smoke run.
6. **Pilot owner:** run A–E, one seed, fixed 15k budget.
7. **Research owner:** freeze main budget after throughput measurement, then run A–D × 3 seeds.
8. **Analysis owner:** use notebook; report per-seed values, AUC, mean/std, bootstrap CI and T0/T1/T2.
9. **Writing owner:** map each claim to a result artifact and state limitations.

## Runtime definition of done

- Targeted tests pass in the real training environment.
- A and D 2k smoke runs finish without NaN/crash.
- Saved checkpoint loads through external evaluator.
- Same evaluation seed yields reproducible start sequence/result protocol.
- TensorBoard export and notebook consume artifacts successfully.
- Tracker exports/imports valid JSON.
- Every result claim points to a manifest and raw artifact.
