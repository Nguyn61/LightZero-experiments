# Research plan — SumToThree Curriculum + Event-Aligned Reward

## 1. Thesis và trạng thái

> Event-aligned reward densification cộng progressive start-state curriculum có thể cải thiện sample efficiency và held-out start-state generalization của Sampled EfficientZero, trong khi success vẫn được đo bằng binary SumToThree objective gốc.

- **Implemented:** intervention code, A–E config factory, tests, kit, scripts.
- **Syntax verified:** Python files compile và JSON templates parse.
- **Runtime verification pending:** cần WSL/Vast.ai environment có `pooltool`, `pytest`, LightZero dependencies.
- **Results pending:** chưa có số để kết luận thesis.

Không claim thuật toán mới, first, SOTA hoặc sim-to-real.

## 2. Frozen problem

- Task: PoolTool SumToThree, two balls, 10 shots/episode.
- Observation: 4-D coordinates.
- Action: normalized 2-D -> cue speed + cut angle; no spin/elevation.
- Primary success: object contact và đúng 3 linear-cushion events.
- Policy: Sampled EfficientZero.
- Frozen parameters: K=20, MCTS=50, replay=1, reanalyze=.25, SSL=2, LR=.003.
- Built-in evaluator: canonical start + binary reward.

## 3. Falsifiable hypotheses

- **H1 — Reward:** B có binary learning-curve AUC/final score cao hơn A.
- **H2 — Curriculum:** C generalize T1/T2 tốt hơn A mà không phá T0.
- **H3 — Interaction:** D tốt nhất trong A–D trên sample efficiency và T0/T1/T2.
- **H4 — Ordering:** nếu D > E, curriculum ordering có giá trị hơn chỉ broad random diversity.
- **H0:** không có effect nhất quán dưới cùng budget/seed protocol.

Failure: shaped collector return tăng nhưng binary score không tăng là reward mismatch, không phải positive result.

## 4. Intervention

### Event-aligned reward

| Outcome | Reward |
| --- | ---: |
| no contact | 0.0 |
| contact + 0/1/2 cushions | 0.1 / 0.2 / 0.3 |
| contact + exactly 3 | 1.0 |
| contact + >3 | 0.1 |

Reward nằm trong `[0,1]`; exact-three là maximum duy nhất. Evaluation không dùng shaped reward.

### Start distributions

- **C0 canonical:** layout gốc.
- **C1 local:** perturb quanh canonical theo 8% usable dimensions.
- **C2 broad:** uniform valid table positions.
- C1/C2 reject overlap/near-overlap và dùng seeded environment RNG.
- Fixed curriculum: 25% C0, 40% C1, 35% C2.

## 5. Factorial experiment

| Arm | Train reward | Train starts | Purpose |
| --- | --- | --- | --- |
| A | binary | canonical | frozen control |
| B | event_aligned | canonical | reward main effect |
| C | binary | curriculum | curriculum main effect |
| D | event_aligned | curriculum | full interaction |
| E optional | event_aligned | broad from start | ordering control |

Protocol machine-readable: [`configs/experiment_matrix.json`](configs/experiment_matrix.json).

## 6. Schedule cho một GPU nhỏ

1. **Correctness:** targeted tests, random diagnostics, A/D 2k smoke.
2. **Pilot:** A–E × seed 0 × 15k steps.
3. **Main:** A–D × seeds 0/1/2 × 40k steps nếu throughput phù hợp.
4. **Ordering control:** E × 3 seeds nếu compute đủ.
5. **Confirmation:** thêm seeds 3/4 cho A và D nếu preliminary effect đáng kể.

Nếu compute thiếu, giảm cùng budget cho mọi arm. Không extend riêng D.

## 7. Evaluation

### Primary

- T0 canonical binary episode score `[0,10]`.
- Binary per-shot success rate.
- Binary learning-curve AUC theo environment steps.

### Secondary

- T1 local và T2 broad binary score.
- Contact rate.
- Cushion-count histogram conditional on contact.
- Action diagnostics nếu rollout collector hỗ trợ.
- Wall time, interactions/sec, NaN/crash/simulator anomaly.

Checkpoint selection dùng canonical validation, không dùng held-out final tests.

## 8. Statistics

- Training seed là independent unit.
- Báo từng seed, mean, standard deviation.
- Bootstrap 95% CI cho difference in seed means.
- Effect size: successful shots per 10-shot episode.
- A–D với 3 seeds là exploratory; headline A-vs-D mạnh hơn nếu có 5 seeds.
- Không biến p-value thành kết luận tự động.

## 9. Execution gates

### Gate 0 — Dependencies

`pooltool`, LightZero, PyTorch, pytest import được; CUDA status được ghi.

### Gate 1 — Tests

Legacy canonical/binary/action behavior và test mới đều pass.

### Gate 2 — Diagnostics

Random policy tạo event distributions hợp lý; exact-three rarity được đo, không đoán.

### Gate 3 — Smoke

A/D 2k runs không crash/NaN; evaluator metric là binary; D đi qua curriculum stage theo config.

### Gate 4 — Artifact validity

Mỗi run có manifest, command, commit, config, seed, budget, logs và checkpoint path/checksum.

### Gate 5 — Main experiment

Chỉ bắt đầu sau khi pilot khóa budget và protocol.

## 10. Decision rules

- Baseline saturation sớm: task canonical không đủ khó; không claim sparse-reward breakthrough.
- Shaped return tăng nhưng binary không tăng: reward hacking/mismatch.
- E >= D: diversity giải thích effect tốt hơn curriculum ordering.
- A-vs-D difference nhỏ hơn practical threshold 0.5 success/episode hoặc CI rất rộng: kết luận negative/inconclusive.
- Run lỗi không bị xóa; đánh dấu failed/excluded với lý do.

## 11. Paper outline

1. Introduction: contact-rich continuous control + exact-event sparse score.
2. Task/baseline: MDP, action mapping, binary objective, fixed-start limitation.
3. Method: event reward, valid sampler, C0/C1/C2 schedule, evaluator isolation.
4. Protocol: frozen SEZ, arms, seeds, budgets, held-out tests, statistics.
5. Results: T0 primary, learning AUC, T1/T2, ablation, diagnostics.
6. Limitations: two balls, assisted aim, no spin/elevation, simulator only, small seeds.
7. Reproducibility appendix: commands, manifests, versions, raw artifacts.

## 12. Next concrete action

Chạy theo thứ tự:

```bash
python -m pytest zoo/pooltool/sum_to_three/envs/test_sum_to_three_env.py -q
python research/sum_to_three_curriculum_reward/scripts/random_policy_diagnostics.py --episodes 100 --output research/sum_to_three_curriculum_reward/records/random-policy-seed0.json
python zoo/pooltool/sum_to_three/config/research_alpha.py --arm A_sparse_fixed --seed 0 --max-env-step 2000
python zoo/pooltool/sum_to_three/config/research_alpha.py --arm D_dense_curriculum --seed 0 --max-env-step 2000
```

Sau mỗi command, tạo/cập nhật manifest và journal entry trước khi chạy bước kế tiếp.
