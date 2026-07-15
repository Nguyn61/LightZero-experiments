# Vì sao chọn LightZero + PoolTool + SumToThree

Tài liệu này giải thích các quyết định nền tảng của dự án để một người đọc (hoặc reviewer) hiểu tại sao stack này hợp lý, và tại sao một stack khác có thể bị hỏi. Mọi claim về code đều có file:line làm bằng chứng. Các claim về đặc tính thuật toán/thư viện bên ngoài được đánh dấu **[cần literature check]** — phải xác minh bằng primary source trước khi đưa vào paper.

---

## 1. Tổng quan quyết định

| Trục | Lựa chọn | Lý do một câu |
|---|---|---|
| Bài toán | Cue sports / billiards | Contact-rich physics + scoring theo sự kiện rời rạc, dễ định nghĩa, khó giải |
| Simulator | PoolTool | Physics engine billiards có event API và ruleset sẵn, deterministic, chạy headless |
| Task cụ thể | SumToThree | Nhỏ nhất trong họ billiards nhưng vẫn giữ đủ độ khó reward-sparse |
| Thuật toán | Sampled EfficientZero (LightZero) | Model-based planning + sampled-action MCTS cho continuous control |
| Observation | Coordinate (4-D) | Loại bỏ confound của representation learning, cô lập biến curriculum/reward |
| Framework | LightZero | Có sẵn MCTS+RL, config/log/checkpoint infra, và integration PoolTool |

Nguyên tắc xuyên suốt: **cố định mọi thứ không phải biến nghiên cứu**, để bất kỳ khác biệt kết quả nào cũng quy được về curriculum/reward chứ không phải representation, thuật toán, hay simulator.

---

## 2. Vì sao là billiards (cue sports)

Billiards là một "sweet spot" hiếm cho RL nghiên cứu vì gộp được nhiều tính chất mà đa số benchmark chỉ có một phần:

1. **Contact-rich, phi tuyến, nhưng deterministic.** Va chạm bi–bi và bi–băng là phi tuyến mạnh và nhạy điều kiện đầu (một thay đổi nhỏ ở góc → quỹ đạo rất khác), nhưng vật lý vẫn xác định. Đây là môi trường lý tưởng để nghiên cứu planning: model phải học một hàm động lực học khó, và MCTS phải tìm được cú đánh tốt trong không gian nhạy cảm.

2. **Scoring theo sự kiện rời rạc trên một action liên tục.** Điểm số được tính từ chuỗi event (chạm bi, chạm băng), không phải từ một hàm reward trơn. Điều này tạo ra một **sparse, discontinuous reward** trên một **continuous action space** — một cấu hình khó cho cả model-free lẫn model-based, và chính là nơi curriculum/reward shaping có đất diễn.

3. **Một shot = một quyết định.** Không giống locomotion (hàng trăm bước nhỏ), mỗi shot là một hành động "đặt cược" duy nhất rồi xem kết quả. Điều này làm bài toán gần với **planning một bước sâu** hơn là control liên tục theo thời gian, khiến nó là testbed sạch cho MCTS.

4. **Trực quan và dễ kể chuyện.** Với một bài student paper, việc reviewer/người đọc nhìn thấy ngay "agent học đánh bi" là một lợi thế truyền thông thật, không phải trang trí.

**Giới hạn cần thành thật:** billiards không phải benchmark chuẩn được cộng đồng dùng rộng rãi, nên không thể so sánh trực tiếp với leaderboard. Đây là điểm yếu về positioning, được xử lý trong [PAPER_GUIDE.md](PAPER_GUIDE.md).

---

## 3. Vì sao là PoolTool (simulator)

PoolTool được chọn thay vì tự viết physics hoặc dùng engine game 3D vì các lý do có thể kiểm chứng trong code:

1. **Event-based physics API sẵn có.** Reward của task được tính trực tiếp từ event stream: `pt.events.filter_type(..., BALL_BALL)` và `pt.events.filter_events(..., BALL_LINEAR_CUSHION, by_ball(...))` tại [`utils.py:114-124`](../../zoo/pooltool/sum_to_three/envs/utils.py). Nghĩa là ta không phải suy diễn "đã chạm chưa" từ toạ độ — simulator trả về sự kiện có ngữ nghĩa. Đây là điều làm reward design (event-aligned reward) khả thi và sạch.

2. **Ruleset và rack dựng sẵn.** `pt.GameType.SUMTOTHREE`, `pt.get_ruleset(...)`, `pt.get_rack(...)`, `pt.Table.from_game_type(...)` tại [`sum_to_three_env.py:214-226`](../../zoo/pooltool/sum_to_three/envs/sum_to_three_env.py) cho phép dựng một ván hợp lệ mà không phải tự định nghĩa luật hay hình học bàn.

3. **Aiming helper.** `pt.aim.at_ball(system, "object", cut=angle)` tại [`sum_to_three_env.py:183`](../../zoo/pooltool/sum_to_three/envs/sum_to_three_env.py) chuyển một "cut angle" tương đối thành góc cơ tuyệt đối. Đây là lý do action chỉ cần 2 chiều (tốc độ + cut) thay vì phải học cả hình học ngắm — hạ độ khó xuống mức một student project.

4. **Deterministic và có safeguard.** `pt.simulate(..., inplace=True, max_events=200)` tại [`datatypes.py:167`](../../zoo/pooltool/datatypes.py) cho physics xác định và chặn được vòng lặp event vô hạn hiếm gặp — quan trọng cho reproducibility.

5. **Chạy headless + có renderer tuỳ chọn.** Coordinate mode không cần đồ hoạ; image mode dùng pygame renderer tạo feature planes `5×20×10` tại [`image_representation.py:406-425`](../../zoo/pooltool/image_representation.py). Nghĩa là cùng một simulator phục vụ cả vector-obs (nhẹ, cho nghiên cứu chính) lẫn visual RL (nặng, cho hướng tương lai) mà không đổi environment.

**Giới hạn cần thành thật:** SumToThree hiện tại chỉ 2 bi, không lỗ, dùng aiming assist, và spin/elevation bị khoá 0 (`theta=a=b=0` tại [`sum_to_three_env.py:186`](../../zoo/pooltool/sum_to_three/envs/sum_to_three_env.py)). Vì vậy nó **không** đại diện cho billiards đầy đủ (pool 9-ball, snooker). Đó là feature của một student-scale project, không phải bug, nhưng phải nói rõ trong Limitations.

---

## 4. Vì sao là SumToThree (task cụ thể)

Trong họ billiards, SumToThree là biến thể **tối giản mà vẫn khó**:

- Luật: điểm = 1 nếu cue chạm object **và** tổng số lần chạm băng của cả hai bi **đúng bằng 3** ([`utils.py:99-142`](../../zoo/pooltool/sum_to_three/envs/utils.py)).
- Chỉ 2 bi, không lỗ, không đối thủ (single-player, `win_condition=-1` tại [`sum_to_three_env.py:145`](../../zoo/pooltool/sum_to_three/envs/sum_to_three_env.py)), episode 10 shots.

Vì sao nó là lựa chọn tốt cho nghiên cứu curriculum/reward:

1. **Reward vừa đủ sparse.** "Đúng 3 băng" là điều kiện hẹp — random policy hiếm khi đạt, nên có khoảng trống rõ để curriculum/shaping cải thiện. Nhưng nó cũng không quá hiếm đến mức không học được gì. (Độ hiếm thực tế phải đo bằng `random_policy_diagnostics.py` trước khi khẳng định — xem [RESEARCH_PLAN.md](RESEARCH_PLAN.md).)
2. **State space nhỏ.** 4 toạ độ → có thể train nhiều seed trên một GPU nhỏ, phù hợp ngân sách student.
3. **Không có đối thủ.** Loại bỏ confound self-play/opponent modeling, cô lập đúng biến ta quan tâm.
4. **"Đúng bằng 3" là mục tiêu phi đơn điệu.** Nhiều băng hơn không tốt hơn — điều này khiến reward shaping thú vị (phải phân biệt "gần đúng" với "vượt quá"), đúng như [thiết kế event_aligned reward](RESEARCH_PLAN.md).

---

## 5. Vì sao là LightZero + Sampled EfficientZero

Đây là quyết định thuật toán quan trọng nhất. Có hai câu hỏi: (a) vì sao model-based planning thay vì model-free, và (b) vì sao chọn LightZero làm framework.

### 5a. Vì sao Sampled EfficientZero (model-based, sampled-action MCTS)

Bài toán có ba đặc điểm khớp thẳng với MuZero-family + sampled action:

1. **Continuous action + planning một bước sâu.** Mỗi shot là một quyết định đắt; giá trị của việc "nhìn trước" (planning) cao. MuZero học một model tiềm ẩn và dùng MCTS để planning — đúng với bản chất "chọn cú đánh tốt nhất" của billiards. **[cần literature check: định vị chính xác so với Sampled MuZero (Hubert et al. 2021).]**

2. **Sampled actions cho không gian liên tục.** MCTS gốc cần action rời rạc; Sampled (Efficient)Zero sample một tập `K` action tại mỗi node để planning trong không gian liên tục. Repo cấu hình `K=20` sampled actions với `sigma_type="conditioned"` (Gaussian có điều kiện) tại [`sum_to_three_vector_obs_sez_config.py:42-46`](../../zoo/pooltool/sum_to_three/config/sum_to_three_vector_obs_sez_config.py). Đây là cơ chế cho phép planning trên action `(V0, cut)` liên tục.

3. **Sample efficiency là biến nghiên cứu.** EfficientZero thêm self-supervised consistency loss (`self_supervised_learning_loss=True` tại [config:48](../../zoo/pooltool/sum_to_three/config/sum_to_three_vector_obs_sez_config.py)) và reanalyze (`reanalyze_ratio=0.25`) để học nhanh hơn với ít dữ liệu — quan trọng khi ngân sách chỉ một GPU nhỏ. Vì sample efficiency chính là thứ curriculum/reward định tác động, dùng một thuật toán mà sample efficiency đã là first-class citizen giúp đo lường sạch hơn.

**So với model-free (SAC).** Repo đã có sẵn một baseline SAC ([`sum_to_three_vector_obs_sac_config.py`](../../zoo/pooltool/sum_to_three/config/sum_to_three_vector_obs_sac_config.py), `cuda=False`, replay buffer `1e6`). SAC là một reference baseline hợp lý, nhưng **không** nên là headline comparison vì: (i) nó chưa được matched-budget với SEZ; (ii) planning không phải điểm mạnh của nó trong bài toán "một shot ăn thua". Vai trò đúng của SAC: một điểm tham chiếu model-free để đặt bối cảnh, được nêu trong [RESEARCH_PLAN.md](RESEARCH_PLAN.md).

### 5b. Vì sao LightZero làm framework

1. **Có sẵn cả họ MCTS+RL.** LightZero cung cấp MuZero, EfficientZero, Sampled EfficientZero, Stochastic/Gumbel MuZero, UniZero — dùng chung entrypoint `train_muzero`/`eval_muzero` ([`eval_muzero.py:42-43`](../../lzero/entry/eval_muzero.py)). Nghĩa là các hướng tương lai (đổi thuật toán mà giữ nguyên env) gần như miễn phí về mặt hạ tầng.

2. **MCTS có C++ tree + Python fallback.** Sampled MCTS có backend ctree ([`mcts_ctree_sampled.py:29-41`](../../lzero/mcts/tree_search/mcts_ctree_sampled.py)) và policy hỗ trợ batched MCTS, Dirichlet noise, PUCT ([`sampled_efficientzero.py:70-100`](../../lzero/policy/sampled_efficientzero.py)). Đây là hạ tầng planning trưởng thành mà một student không nên tự viết lại.

3. **Replay buffer có priority + reanalysis.** [`game_buffer_sampled_efficientzero.py:48-113`](../../lzero/mcts/buffer/game_buffer_sampled_efficientzero.py) — cần thiết cho reanalyze, và là extension point cho các hướng tương lai (prioritized/curriculum-aware replay).

4. **Config/log/checkpoint convention chuẩn.** `train_muzero` ghi TensorBoard vào `<exp_name>/log/serial`, checkpoint vào `<exp_name>/ckpt` ([`train_muzero.py:84-115`](../../lzero/entry/train_muzero.py)). Kit này bám đúng convention đó, nên artifacts reproducible mà không phải dựng lại pipeline.

5. **Đã có sẵn integration PoolTool.** Đây là lý do thực dụng nhất: environment `pooltool_sumtothree` đã được đăng ký và chạy được, nên công sức dồn vào **nghiên cứu** (curriculum/reward) thay vì **plumbing**.

**Giới hạn cần thành thật:** LightZero/SEZ là hệ phức tạp; nhiều siêu tham số bị khoá (đúng ý đồ) nghĩa là ta không khám phá không gian thuật toán. Và Cython extension chủ yếu cho Linux (đã thấy trong replicate_log — phải chuyển sang WSL). Đây là chi phí vận hành thật.

---

## 6. Vì sao coordinate observation (không phải image) cho nghiên cứu chính

- Coordinate obs là 4 số `[cue_x, cue_y, object_x, object_y]` ([`utils.py:74-92`](../../zoo/pooltool/sum_to_three/envs/utils.py)). Nó **đủ** để mô tả trạng thái đầy đủ của task (2 bi tĩnh).
- Dùng vector obs **loại bỏ confound representation learning**: nếu dùng image, một cải thiện có thể đến từ CNN học biểu diễn tốt hơn chứ không phải từ curriculum/reward. Vì biến nghiên cứu là curriculum/reward, ta phải cô lập chúng.
- Vector obs rẻ hơn nhiều → nhiều seed hơn trên một GPU nhỏ.

Image mode (`5×20×10` feature planes) vẫn tồn tại và là một **hướng tương lai** riêng (visual RL / robustness), không phải biến của nghiên cứu hiện tại. Xem [FUTURE_THESES.md](FUTURE_THESES.md).

---

## 7. Tóm tắt cho phần Methods của paper

Khi viết, đóng khung như sau (một đoạn):

> Chúng tôi nghiên cứu trên PoolTool SumToThree, một task billiards single-player với physics dựa trên sự kiện và scoring thưa (cue chạm object và đúng ba lần chạm băng). Trạng thái được biểu diễn bằng toạ độ hai bi (4-D) và hành động là vector liên tục hai chiều (tốc độ cơ và cut angle, với ngắm được hỗ trợ bởi simulator; spin và elevation cố định bằng 0). Chúng tôi dùng Sampled EfficientZero (LightZero) vì bài toán kết hợp không gian hành động liên tục với planning một-bước-sâu, nơi sampled-action MCTS và consistency-based sample efficiency là phù hợp. Chúng tôi cố định toàn bộ siêu tham số thuật toán và biểu diễn để cô lập tác động của thiết kế reward và curriculum khởi tạo, và đánh giá mọi arm dưới hàm reward nhị phân gốc.

Mọi con số/đặc tính cụ thể trong đoạn trên đều truy được về file:line ở các mục trên; các phát biểu về vị trí học thuật của Sampled MuZero/EfficientZero cần **[literature check]** trước submission.
