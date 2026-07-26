# MỤC LỤC — Bắt đầu từ đây

File này là điểm vào duy nhất của kit. Nó trả lời hai câu hỏi: **có gì trong này** và **làm theo thứ tự nào**.

Vị trí kit trong repo: `research/sum_to_three_curriculum_reward/`
Mở nhanh: đọc file này, rồi mở [`README.md`](README.md), hoặc double-click [`progress_tracker.html`](progress_tracker.html).

---

## 1. Có gì trong kit (bảng tra cứu)

| File | Loại | Khi nào dùng |
|---|---|---|
| [`INDEX.md`](INDEX.md) | Điều hướng | Bạn đang đọc — điểm bắt đầu |
| [`README.md`](README.md) | Điều hướng | Tổng quan, quick start, trạng thái |
| [`WHY_LIGHTZERO_POOLTOOL.md`](WHY_LIGHTZERO_POOLTOOL.md) | Lý do | Hiểu vì sao chọn LightZero/PoolTool/SumToThree |
| [`HUMAN_GUIDE.md`](HUMAN_GUIDE.md) | Hướng dẫn | Hiểu thiết kế nghiên cứu (dành cho người đã biết RL) |
| [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) | Protocol | Hypotheses, arms A–E, metrics, decision gates |
| [`progress_tracker.html`](progress_tracker.html) | Công cụ | Theo dõi tiến độ, ghi run, export/import JSON (offline) |
| [`notebooks/experiment_lab.ipynb`](notebooks/experiment_lab.ipynb) | Công cụ | Diagnostics, learning curve, AUC, CI, so sánh, journal |
| [`JOURNAL_GUIDE.md`](JOURNAL_GUIDE.md) | Hướng dẫn | Cách ghi journal đúng chuẩn evidence |
| [`PAPER_GUIDE.md`](PAPER_GUIDE.md) | Hướng dẫn | Positioning và viết paper |
| [`FUTURE_THESES.md`](FUTURE_THESES.md) | Tham khảo | Hướng nghiên cứu/tối ưu tương lai |
| [`AGENT_CONTEXT.md`](AGENT_CONTEXT.md) | Handoff | Context để một agent khác tiếp tục từ đầu |
| [`configs/experiment_matrix.json`](configs/experiment_matrix.json) | Protocol | Nguồn machine-readable của arms/frozen policy |
| [`configs/README.md`](configs/README.md) | Hướng dẫn | Command để chạy từng arm |
| [`scripts/`](scripts/) | Code | Diagnostics, TensorBoard export, checkpoint eval |
| [`templates/`](templates/) | Mẫu | Manifest, result summary, experiment entry |
| [`records/`](records/README.md) | Dữ liệu | Nơi lưu hồ sơ từng run |

Code runtime thực tế nằm ngoài kit, ở vị trí chuẩn LightZero:
- [`../../zoo/pooltool/sum_to_three/envs/utils.py`](../../zoo/pooltool/sum_to_three/envs/utils.py) — reward + shot outcome
- [`../../zoo/pooltool/sum_to_three/envs/sum_to_three_env.py`](../../zoo/pooltool/sum_to_three/envs/sum_to_three_env.py) — env, start sampler, curriculum
- [`../../zoo/pooltool/sum_to_three/config/research_alpha.py`](../../zoo/pooltool/sum_to_three/config/research_alpha.py) — arm configs

---

## 2. Làm theo thứ tự này

### Giai đoạn 0 — Hiểu (đọc, chưa chạy gì)

1. Đọc [`README.md`](README.md).
2. Đọc [`WHY_LIGHTZERO_POOLTOOL.md`](WHY_LIGHTZERO_POOLTOOL.md) — hiểu vì sao stack này.
3. Đọc [`HUMAN_GUIDE.md`](HUMAN_GUIDE.md) và [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) — hiểu thí nghiệm.

### Giai đoạn 1 — Chuẩn bị môi trường (WSL/Vast.ai)

4. Kích hoạt Python 3.10 env đã cài LightZero (`pip install -e .`).
5. Chạy tests:
   ```bash
   python -m pytest zoo/pooltool/sum_to_three/envs/test_sum_to_three_env.py -q
   ```

### Giai đoạn 2 — Đo độ khó (diagnostics)

6. Chạy random-policy diagnostics:
   ```bash
   python research/sum_to_three_curriculum_reward/scripts/random_policy_diagnostics.py \
     --episodes 100 \
     --output research/sum_to_three_curriculum_reward/records/random-policy-seed0.json
   ```

### Giai đoạn 3 — Smoke (2k steps, kiểm tra pipeline)

7. Baseline và full-method:
   ```bash
   python zoo/pooltool/sum_to_three/config/research_alpha.py --arm A_sparse_fixed --seed 0 --max-env-step 2000 --attempt 01
   python zoo/pooltool/sum_to_three/config/research_alpha.py --arm D_dense_curriculum --seed 0 --max-env-step 2000 --attempt 01
   ```

### Giai đoạn 4 — Ghi hồ sơ

8. Mở [`progress_tracker.html`](progress_tracker.html), tạo run record cho mỗi run.
9. Copy [`templates/run_manifest.json`](templates/run_manifest.json) và [`templates/experiment_entry.md`](templates/experiment_entry.md) vào `records/<run-id>/`.

### Giai đoạn 5 — Chạy chính (pilot → main)

10. Pilot A–E, 1 seed, 15k–20k steps. Khoá budget.
11. Main A–D, 3 seeds, cùng budget. Xem [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) §6.

### Giai đoạn 6 — Phân tích

12. Export TensorBoard + evaluate checkpoint (xem [`configs/README.md`](configs/README.md)).
13. Mở [`notebooks/experiment_lab.ipynb`](notebooks/experiment_lab.ipynb), nạp JSON, tạo learning curve/AUC/CI/T0-T1-T2.

### Giai đoạn 7 — Viết

14. Ghi journal theo [`JOURNAL_GUIDE.md`](JOURNAL_GUIDE.md).
15. Viết paper theo [`PAPER_GUIDE.md`](PAPER_GUIDE.md).
16. Muốn mở rộng tiếp: [`FUTURE_THESES.md`](FUTURE_THESES.md).

---

## 3. Nguyên tắc vàng (đừng quên)

- Headline metric luôn là **binary SumToThree score** — không dùng shaped reward để tuyên bố thành công.
- Không overwrite run cũ; mỗi run có `attempt` riêng.
- Không giấu run fail/excluded.
- Không đổi budget/metric sau khi nhìn kết quả.
- Không claim "novel/first/SOTA" trước literature review primary-source.
- Synthetic data trong notebook chỉ để test pipeline, **không phải kết quả**.

---

## 4. Nếu bạn là một agent tiếp nhận

Đọc [`AGENT_CONTEXT.md`](AGENT_CONTEXT.md) trước tiên — nó có trạng thái đã verify, guardrails, và next-task queue để bắt đầu từ đầu mà không nhầm planned với implemented.
