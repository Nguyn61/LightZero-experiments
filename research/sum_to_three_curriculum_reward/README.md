# Student Research Kit: SumToThree Curriculum + Reward

Một bộ tự chứa cho học sinh đã biết RL: research context, protocol, code entrypoints, HTML tracker, Jupyter notebook, journal guide và normalized artifacts.

> **Bắt đầu ở đâu?** Mở [`INDEX.md`](INDEX.md) — mục lục toàn kit kèm hướng dẫn làm theo thứ tự.

## Research question

> Event-aligned reward densification cộng progressive start-state curriculum có cải thiện binary sample efficiency và generalization của Sampled EfficientZero trên SumToThree không?

Primary metric luôn là **binary SumToThree score gốc**. Shaped return chỉ là diagnostic.

## Trạng thái

| Hạng mục | Trạng thái |
| --- | --- |
| Reward `event_aligned`, valid starts, curriculum, evaluator isolation | **Implemented; chưa runtime verified trong environment đầy đủ** |
| Arm A–E config factory và diagnostics/evaluation scripts | **Implemented; đã syntax-check** |
| Documentation, tracker, notebook, templates | **Implemented** |
| Pytest + A/D training smoke | **Blocked trong Windows Python hiện tại: thiếu `pytest` và `pooltool`** |
| Quantitative results/paper claims | **Planned** |

Đọc [`AGENT_CONTEXT.md`](AGENT_CONTEXT.md) để biết chính xác phần nào là code, phần nào đã verify và next tasks.

## Quick start 10 phút

Từ repository root, trong WSL/Vast.ai environment đã cài dependencies:

```bash
source venv/bin/activate
python -m pytest zoo/pooltool/sum_to_three/envs/test_sum_to_three_env.py -q
```

Random-policy event diagnostic:

```bash
python research/sum_to_three_curriculum_reward/scripts/random_policy_diagnostics.py \
  --episodes 100 \
  --output research/sum_to_three_curriculum_reward/records/random-policy-seed0.json
```

Baseline smoke:

```bash
python zoo/pooltool/sum_to_three/config/research_alpha.py \
  --arm A_sparse_fixed --seed 0 --max-env-step 2000
```

Full-method smoke:

```bash
python zoo/pooltool/sum_to_three/config/research_alpha.py \
  --arm D_dense_curriculum --seed 0 --max-env-step 2000
```

Mở tracker bằng double-click [`progress_tracker.html`](progress_tracker.html). Mở lab:

```bash
jupyter notebook research/sum_to_three_curriculum_reward/notebooks/experiment_lab.ipynb
```

## Thứ tự đọc

1. [`WHY_LIGHTZERO_POOLTOOL.md`](WHY_LIGHTZERO_POOLTOOL.md) — vì sao chọn stack này (LightZero, PoolTool, SumToThree).
2. [`HUMAN_GUIDE.md`](HUMAN_GUIDE.md) — hiểu thiết kế nghiên cứu.
3. [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) — hypotheses, experiment matrix và decision gates.
4. [`notebooks/experiment_lab.ipynb`](notebooks/experiment_lab.ipynb) — diagnostics và analysis.
5. [`JOURNAL_GUIDE.md`](JOURNAL_GUIDE.md) — ghi evidence đúng cách.
6. [`PAPER_GUIDE.md`](PAPER_GUIDE.md) — kể chuyện và positioning cho paper.
7. [`FUTURE_THESES.md`](FUTURE_THESES.md) — các hướng tối ưu/mở rộng tương lai.
8. [`AGENT_CONTEXT.md`](AGENT_CONTEXT.md) — handoff cho future agent.

## Folder map

```text
research/sum_to_three_curriculum_reward/
├── INDEX.md                     # mục lục + hướng dẫn theo thứ tự
├── README.md                    # entry point
├── WHY_LIGHTZERO_POOLTOOL.md    # rationale cho stack
├── AGENT_CONTEXT.md             # machine/handoff context
├── RESEARCH_PLAN.md             # protocol khoa học
├── HUMAN_GUIDE.md               # giải thích cho học sinh
├── JOURNAL_GUIDE.md             # cách viết journal/report
├── PAPER_GUIDE.md               # positioning và viết paper
├── FUTURE_THESES.md             # hướng nghiên cứu tương lai
├── progress_tracker.html        # offline tracker + localStorage
├── configs/
│   ├── experiment_matrix.json
│   └── README.md
├── notebooks/
│   └── experiment_lab.ipynb
├── scripts/
│   ├── random_policy_diagnostics.py
│   ├── export_tensorboard.py
│   └── evaluate_checkpoint.py
├── templates/
│   ├── experiment_entry.md
│   ├── run_manifest.json
│   └── result_summary.json
└── records/
    └── README.md
```

Actual runtime code vẫn ở vị trí chuẩn của LightZero:

- [`../../zoo/pooltool/sum_to_three/envs/utils.py`](../../zoo/pooltool/sum_to_three/envs/utils.py)
- [`../../zoo/pooltool/sum_to_three/envs/sum_to_three_env.py`](../../zoo/pooltool/sum_to_three/envs/sum_to_three_env.py)
- [`../../zoo/pooltool/sum_to_three/config/research_alpha.py`](../../zoo/pooltool/sum_to_three/config/research_alpha.py)

## Evidence workflow

```text
experiment_matrix.json
  -> train arm/seed
  -> raw TensorBoard + checkpoint
  -> run manifest + journal entry
  -> binary T0/T1/T2 evaluation JSON
  -> notebook analysis
  -> result summary + report claim
```

Tracker là công cụ quản lý cá nhân, không phải scientific source of truth. Manifest, raw log, checkpoint checksum và evaluation JSON mới là evidence.

## Quy tắc vàng

- Không overwrite run cũ.
- Không giấu run fail/excluded.
- Không thay budget/metric sau khi nhìn kết quả.
- Không dùng shaped reward để tuyên bố task gốc tốt hơn.
- Không claim “novel/first/SOTA” trước literature review primary-source.
