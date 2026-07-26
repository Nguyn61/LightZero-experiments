# Danh mục thesis tương lai (để tối ưu và mở rộng)

Danh mục các hướng nghiên cứu tiếp theo, xếp theo **tier rủi ro/novelty**. Mỗi thesis có: giả thuyết, extension point trong code (file:line), thí nghiệm tối thiểu, chi phí compute, rủi ro novelty, và decision gate.

Quy ước tin cậy:
- **Ý tưởng chắc chắn:** buildable ngay với extension point đã tồn tại; kết quả có ý nghĩa dù dương hay âm.
- **Giả thuyết:** hợp lý nhưng cần pilot để biết có effect không.
- **Rủi ro cao:** novelty tiềm năng lớn nhưng dễ thất bại hoặc trùng literature.

Nguyên tắc chung: chỉ đổi **một trục** mỗi thesis; giữ nguyên phần còn lại của [protocol đã khoá](RESEARCH_PLAN.md); headline metric luôn là binary SumToThree score.

---

## Tier 1 — Student-scale, chắc chắn (một GPU nhỏ)

### T1.1 — Reward shaping ablation sâu hơn (Ý tưởng chắc chắn)

**Giả thuyết:** Hình dạng của dense reward (không chỉ có/không) quyết định mức cải thiện; một số reward "gần đúng bằng 3" tốt hơn linear milestone.

**Extension point:** reward registry `_reward_functions` tại [`utils.py`](../../zoo/pooltool/sum_to_three/envs/utils.py) — thêm biến thể mới cạnh `binary`/`event_aligned` là thêm một entry + một hàm `*_from_outcome`.

**Thí nghiệm:** so 3–4 reward shapes (linear milestone hiện tại; distance-to-3 penalty; potential-based shaping thật; asymmetric penalty cho >3) trên cùng arm fixed-start, 3 seeds. Đây là ablation reward thuần.

**Compute:** ~ như một arm hiện tại × số biến thể. Rẻ.

**Novelty:** thấp về thuật toán; giá trị là **empirical characterization** của reward landscape cho một task sparse cụ thể. **[literature check: potential-based shaping — Ng, Harada, Russell 1999.]**

**Decision gate:** nếu mọi shape cải thiện như nhau → reward chỉ cần "có tín hiệu", hình dạng không quan trọng (kết quả âm vẫn đáng viết).

---

### T1.2 — Competence-based (adaptive) curriculum (Ý tưởng chắc chắn → Giả thuyết)

**Giả thuyết:** Curriculum tự điều chỉnh theo success rate thực (mở rộng phân phối khi agent đạt ngưỡng) tốt hơn fixed schedule 25/40/35.

**Extension point:** logic stage hiện ở [`sum_to_three_env.py` `_resolve_start_distribution`](../../zoo/pooltool/sum_to_three/envs/sum_to_three_env.py) dùng `_reset_count`. Adaptive version thay `progress` bằng một tín hiệu success trượt. **Cảnh báo:** cần đồng bộ success qua các subprocess env (khó hơn), nên đây là bước tăng độ khó có chủ đích.

**Thí nghiệm:** so fixed schedule (D hiện tại) vs adaptive threshold curriculum, 3 seeds, cùng budget.

**Compute:** trung bình.

**Novelty:** thấp-trung; curriculum learning là established, nhưng adaptive-vs-fixed trên một sparse billiards task là empirical đóng góp sạch. **[literature check: teacher-student / automatic curriculum surveys.]**

**Decision gate:** nếu adaptive không hơn fixed → báo rằng ordering đơn giản là đủ; tránh phức tạp hoá.

---

### T1.3 — Sample-efficiency knob sweep có kiểm soát (Ý tưởng chắc chắn)

**Giả thuyết:** Với reward/curriculum tốt nhất (arm D), các knob sample-efficiency (`reanalyze_ratio`, `ssl_loss_weight`, `num_simulations`) có tương tác — ví dụ reanalyze giúp nhiều hơn khi reward đã dense.

**Extension point:** đã có sẵn các exp1–exp6 one-factor configs; và `build_research_config` trong [`research_alpha.py`](../../zoo/pooltool/sum_to_three/config/research_alpha.py) đọc `frozen_policy` từ [`experiment_matrix.json`](configs/experiment_matrix.json). Sweep = nới một field khỏi "frozen".

**Thí nghiệm:** với arm D cố định, sweep từng knob một quanh baseline, 2 seeds pilot rồi 3 seeds cho knob hứa hẹn.

**Compute:** trung bình-cao (nhiều cell). Dùng successive halving.

**Novelty:** thấp; giá trị là hiểu **interaction** giữa reward density và planning budget.

**Decision gate:** đây là hướng tuning, chỉ làm sau khi câu chuyện curriculum/reward chính đã ổn; không để nó thành grab-bag hyperparameter.

---

## Tier 2 — Trung bình (cần nhiều compute hoặc code hơn)

### T2.1 — Spin và elevation: mở rộng action space (Giả thuyết)

**Giả thuyết:** Cho agent điều khiển spin (`a`, `b`) và/hoặc elevation (`theta`) tạo bài toán giàu kỹ năng hơn, và curriculum trở nên quan trọng hơn vì không gian hành động lớn hơn.

**Extension point:** hiện `set_action` khoá `theta=a=b=0` tại [`sum_to_three_env.py:186`](../../zoo/pooltool/sum_to_three/envs/sum_to_three_env.py); action space là 2-D tại [`get_action_space`](../../zoo/pooltool/sum_to_three/envs/sum_to_three_env.py). Mở rộng = nới các tham số này thành action dims và cập nhật `action_space_size`.

**Thí nghiệm:** so 2-D action (hiện tại) vs 3-D (+spin) vs 4-D, đo sample efficiency và ceiling. Curriculum có giúp nhiều hơn khi action lớn hơn không?

**Compute:** cao (action lớn hơn → K sampled actions cần xem lại, học chậm hơn).

**Novelty:** trung; "action-space expansion + curriculum" là một câu chuyện thí nghiệm hợp lý.

**Rủi ro:** spin làm physics khó hơn nhiều; có thể agent không học được trong budget nhỏ. Pilot bắt buộc trước khi cam kết.

---

### T2.2 — Visual RL và robustness (Giả thuyết → Rủi ro cao)

**Giả thuyết:** Trên image observation (`5×20×10` feature planes), curriculum/reward vẫn hiệu quả; và học từ ảnh có robustness khác vector obs.

**Extension point:** image mode đã có ([`image_representation.py:406-425`](../../zoo/pooltool/image_representation.py)), config image SEZ tồn tại; `RenderConfig`/`RenderPlane` cho phép ablate feature planes ([`image_representation.py:108-202`](../../zoo/pooltool/image_representation.py)).

**Thí nghiệm:** lặp lại A–D trên image obs; ablate feature planes (bỏ "joining line", bỏ cushion planes); đo generalization.

**Compute:** cao (CNN + image budget `1e6` steps).

**Novelty:** trung; nhưng cẩn thận vì đây là hai biến (representation + curriculum) — dễ mất tính cô lập.

**Decision gate:** chỉ làm sau khi vector-obs story hoàn chỉnh; nếu không sẽ thành unfocused.

---

### T2.3 — Thuật toán swap có kiểm soát (Ý tưởng chắc chắn về infra, Giả thuyết về kết quả)

**Giả thuyết:** Câu chuyện curriculum/reward **transfer** sang thuật toán khác (MuZero, EfficientZero, Gumbel/Stochastic MuZero, UniZero) — tức là nó là tính chất của task chứ không phải artifact của Sampled EfficientZero.

**Extension point:** entrypoint chung hỗ trợ tất cả các thuật toán này ([`eval_muzero.py:42-43`](../../lzero/entry/eval_muzero.py)); chỉ đổi `create_config.policy.type` + import.

**Thí nghiệm:** chạy arm A và D trên 2 thuật toán, kiểm tra dấu của effect có giữ nguyên không.

**Compute:** cao (nhân đôi số run).

**Novelty:** trung; "robustness của một training-protocol effect qua các thuật toán planning" là một đóng góp phương pháp luận đáng giá.

---

## Tier 3 — Rủi ro cao, novelty tiềm năng lớn

### T3.1 — Physics-informed sampled-action MCTS (Rủi ro cao)

**Giả thuyết:** Dùng cấu trúc vật lý (candidate shots gợi ý bởi hình học ngắm) để **bias phân phối sample action** tại root MCTS cải thiện sample efficiency so với Gaussian conditioned.

**Extension point:** sampled action sinh ở policy/buffer ([`game_buffer_sampled_efficientzero.py:48-113`](../../lzero/mcts/buffer/game_buffer_sampled_efficientzero.py), [`sampled_efficientzero.py:70-100`](../../lzero/policy/sampled_efficientzero.py)). Đây là **thay đổi thuật toán** — vi phạm ràng buộc "không sửa MCTS/policy" của project hiện tại, nên là một project riêng.

**Thí nghiệm:** so Gaussian sampling vs physics-biased sampling, cùng K, cùng budget.

**Compute:** cao; và risk kỹ thuật cao (dễ làm hỏng planning).

**Novelty:** **cao nếu thành công** — "domain-structured action sampling for continuous MCTS" có thể là đóng góp thuật toán thật. **[literature check bắt buộc: progressive widening, action abstraction trong continuous MCTS.]**

**Rủi ro:** dễ trùng ý tưởng đã có; dễ không cải thiện; cần verify novelty kỹ trước khi đầu tư.

---

### T3.2 — Learned world-model quality vs planning depth (Rủi ro cao)

**Giả thuyết:** Trong task contact-rich này, sai số của learned dynamics model tăng nhanh theo unroll depth, nên tăng `num_unroll_steps` không giúp (hoặc hại) — và có thể đo trực tiếp model error trên physics thật.

**Extension point:** so sánh latent model prediction với PoolTool ground-truth (simulator có sẵn, deterministic). `num_unroll_steps=3` hiện tại tại [config](../../zoo/pooltool/sum_to_three/config/sum_to_three_vector_obs_sez_config.py).

**Thí nghiệm:** đo model prediction error theo depth; sweep `num_unroll_steps`; tương quan với performance.

**Compute:** trung-cao.

**Novelty:** cao; "model fidelity vs planning horizon trong contact-rich control" là một câu hỏi khoa học thật, và billiards là testbed sạch vì có ground-truth simulator.

---

### T3.3 — Mở rộng sang billiards đầy đủ (Rủi ro cao, dài hạn)

**Giả thuyết:** Protocol curriculum/reward scale lên các game khó hơn (có lỗ, nhiều bi, snooker/9-ball) — hướng README PoolTool đã bỏ ngỏ ("What billiards game would you like to see next?").

**Extension point:** cần env mới trong `zoo/pooltool/` (two-player, MCTSBot/RuleBot hiện là `NotImplementedError` tại [`sum_to_three_env.py:9-17`](../../zoo/pooltool/sum_to_three/envs/sum_to_three_env.py)).

**Compute:** rất cao. Novelty cao nhưng scope vượt student project — chỉ ghi lại như tầm nhìn dài hạn.

---

## Ma trận ưu tiên gợi ý

| Thesis | Novelty | Compute | Rủi ro | Nên làm khi |
|---|---|---|---|---|
| T1.1 Reward shapes | Thấp | Rẻ | Thấp | Ngay sau nghiên cứu chính |
| T1.2 Adaptive curriculum | Thấp-TB | TB | Thấp | Sau T1.1 |
| T1.3 Efficiency knobs | Thấp | TB-Cao | Thấp | Khi story chính ổn |
| T2.1 Spin/elevation | TB | Cao | TB | Muốn task giàu hơn |
| T2.2 Visual RL | TB | Cao | TB-Cao | Sau vector-obs story |
| T2.3 Algorithm swap | TB | Cao | Thấp | Muốn robustness claim |
| T3.1 Physics-informed MCTS | Cao | Cao | Cao | Có thời gian + literature review |
| T3.2 Model fidelity vs depth | Cao | TB-Cao | Cao | Muốn đóng góp khoa học sâu |
| T3.3 Full billiards | Cao | Rất cao | Cao | Dài hạn / nhóm |

**Khuyến nghị lộ trình:** hoàn tất nghiên cứu curriculum/reward hiện tại → T1.1 → T1.2 → (chọn một trong T2.3 hoặc T3.2 tuỳ mục tiêu paper). T3.1 là "moonshot" chỉ theo đuổi khi đã có literature review khẳng định novelty.
