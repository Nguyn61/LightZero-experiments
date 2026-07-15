# Hướng dẫn viết paper

Tài liệu này bổ sung cho [JOURNAL_GUIDE.md](JOURNAL_GUIDE.md) (cách ghi evidence) bằng cách nói về **cách kể câu chuyện** và **positioning** cho một bài student paper. Nó giả định bạn đã có kết quả thật (không phải synthetic demo).

---

## 1. Chọn tier đóng góp trước khi viết

Kết quả của bạn sẽ rơi vào một trong ba tier. Xác định tier **trước** khi viết để không over-claim.

| Tier | Điều kiện | Câu chuyện |
|---|---|---|
| **A. Positive empirical** | Arm D thắng A trên binary T0 với CI dương và effect ≥ ngưỡng thực dụng (0.5 success/episode) | "Một protocol reward+curriculum đơn giản cải thiện sample efficiency và generalization trên một task billiards sparse" |
| **B. Partial / nuanced** | Một số arm thắng, một số không; hoặc thắng T0 nhưng không T1/T2 | "Reward density giúp nhưng curriculum ordering không; hoặc ngược lại" — vẫn là đóng góp thật |
| **C. Negative / reproducibility** | Không arm nào thắng rõ, hoặc task quá dễ | "Một reproducibility study cho thấy protocol X không giúp trên task Y, và vì sao" |

Cả ba tier đều publishable ở mức student/workshop nếu protocol sạch. **Negative result với protocol nghiêm túc tốt hơn positive result với protocol lỏng.**

---

## 2. Positioning: định vị đóng góp cho đúng

Đây là phần dễ bị reviewer đánh nhất. Ba quy tắc:

1. **Không claim thuật toán mới.** Curriculum learning, reward shaping, Sampled EfficientZero đều đã có. Đóng góp của bạn là **một empirical training protocol và bằng chứng cho nó trên một task cụ thể** — không hơn.

2. **Không claim "first" hay "SOTA".** Trước khi dùng bất kỳ từ nào trong {"novel", "first", "state-of-the-art"}, bạn phải hoàn thành một literature review truy cập được full-text về: cue-sports RL, sparse-reward RL, curriculum learning, potential-based reward shaping, Sampled MuZero/EfficientZero. (Workflow web tự động trong dự án này **chưa** trích xuất được claim đáng tin — phải làm thủ công.)

3. **Đóng khung là case study.** Ngôn ngữ an toàn: "we study", "we find", "we evaluate", "in this environment". Tránh "we prove", "we outperform prior billiards RL".

---

## 3. Kết cấu paper (section-by-section)

### Abstract
- 1 câu bối cảnh (billiards = continuous action + sparse event scoring).
- 1 câu điều bạn làm (event-aligned reward + start curriculum trên SumToThree).
- 1 câu cách đánh giá (mọi kết luận dưới binary reward gốc, N seeds).
- 1 câu kết quả **có số** (Δ success/episode và CI) — điền sau khi có data.
- 1 câu limitation (2 bi, assisted aim, simulator-only).

### 1. Introduction
- Vì sao sparse-reward continuous control khó.
- Vì sao billiards là testbed tốt (rút gọn từ [WHY_LIGHTZERO_POOLTOOL.md](WHY_LIGHTZERO_POOLTOOL.md) §2).
- Phát biểu đóng góp chính xác, hẹp.

### 2. Task and baseline
- SumToThree MDP: obs 4-D, action 2-D (`V0`, cut), episode 10 shots, binary reward (đúng 3 băng).
- Vì sao aiming assist + spin=0 (đơn giản có chủ đích).
- Frozen Sampled EfficientZero + siêu tham số.
- Hạn chế fixed-start của baseline → motivation cho curriculum.
- Trích ngắn từ [WHY_LIGHTZERO_POOLTOOL.md](WHY_LIGHTZERO_POOLTOOL.md) §3–§6.

### 3. Method
- Event-aligned reward (bảng lookup, nhấn "exact-3 là maximum duy nhất").
- Valid randomized start sampler (reject overlap, seeded).
- C0→C1→C2 schedule.
- **Train/eval separation** — nhấn mạnh: collector dùng treatment, evaluator luôn binary+canonical. Đây là điểm phương pháp luận đáng khoe.

### 4. Experimental protocol
- Frozen hyperparameters (bảng).
- Arms A–E (bảng 2×2 + control).
- Seeds, budgets, checkpoint selection, T0/T1/T2 splits.
- Statistics: per-seed, mean±std, bootstrap CI, effect size.
- Đây là section chứng minh bạn không cherry-pick.

### 5. Results
- **Primary first:** binary T0 score, A–D. Learning curve + bảng.
- Sample efficiency (AUC).
- Generalization (T1/T2).
- Ablation (A vs B vs C vs D tách reward khỏi curriculum).
- Control E (ordering vs diversity).
- Event diagnostics (cushion histogram) — giải thích *vì sao* effect xảy ra.

### 6. Discussion & Limitations
- Reward densification **không** phải policy-invariant → có thể đổi policy; đó là lý do chỉ báo binary metric.
- Limitations: 2 bi, assisted aim, spin=0, số seed nhỏ, simulator-only, không transfer sang bàn thật.

### 7. Reproducibility appendix
- Commands, manifests, seeds, versions, hardware.
- Link tới kit này.

---

## 4. Figures và tables (checklist)

| # | Loại | Nội dung |
|---|---|---|
| Fig 1 | Schematic | Canonical start + action `(V0, cut)` + mục tiêu "đúng 3 băng" |
| Fig 2 | Method diagram | C0→C1→C2 + bảng event-aligned reward |
| Fig 3 | Learning curves | Binary T0 return vs env steps, band theo seed |
| Table 1 | Generalization | T0/T1/T2 binary score cho A–E |
| Table 2 | Ablation | A/B/C/D + effect size + CI |
| Fig 4 | Diagnostics | Cushion-count histogram theo start distribution |
| Table 3 | Efficiency | Interactions, wall-clock, hardware |
| Fig 5 | Qualitative | 1 shot thành công, 1 thất bại >3, 1 off-distribution |

Notebook [`experiment_lab.ipynb`](notebooks/experiment_lab.ipynb) đã sinh được phần lớn các số này.

---

## 5. Claim–evidence discipline

Mỗi câu trong Results phải trỏ về một artifact. Dùng bảng nội bộ (không in vào paper) trước khi viết:

| Claim | Metric | Comparison | Artifact (record path) | Trạng thái |
|---|---|---|---|---|
| "D cải thiện sample efficiency" | binary AUC | D vs A | records/... | pending |
| "curriculum giúp generalization" | T2 binary | D vs B | records/... | pending |

Nếu một claim không có dòng ở đây → cắt claim, không cắt evidence.

---

## 6. Venue và độ dài

- **Workshop / student track** là đích thực tế cho tier A/B/C (4–8 trang).
- Nếu kết quả mạnh và literature review khẳng định gap → có thể nhắm venue lớn hơn, nhưng cần thêm seeds (5+) và có thể thêm T2.3 (algorithm robustness) để tăng sức thuyết phục.

---

## 7. Tiêu đề và abstract (bản nháp để chọn)

Các phương án tiêu đề (chọn theo tier kết quả):

- Tier A: *"Event-Aligned Rewards and Start-State Curricula Improve Sample Efficiency on a Sparse Billiards Task"*
- Tier B: *"When Does Curriculum Help? An Ablation of Reward Density and Start-State Expansion on SumToThree"*
- Tier C: *"A Reproducibility Study of Reward Shaping and Curricula for Sparse Continuous Billiards Control"*

Không chốt tiêu đề trước khi có data; tiêu đề phải phản ánh đúng kết quả thật.

---

## 8. Trước khi submit — cổng cuối

- [ ] Literature review full-text hoàn tất; mọi từ novelty có citation.
- [ ] Mọi claim có dòng trong claim–evidence matrix.
- [ ] Mọi số trong abstract trỏ về result summary.
- [ ] Limitations trung thực (danh sách ở §3.6).
- [ ] Reproducibility appendix đầy đủ command/seed/version.
- [ ] Không dùng shaped return làm headline ở bất kỳ đâu.
- [ ] Failed/excluded runs được báo cáo, không giấu.
