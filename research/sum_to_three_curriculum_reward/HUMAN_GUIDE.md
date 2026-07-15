# Human guide — làm nghiên cứu SumToThree RL

Tài liệu dành cho học sinh đã biết khái niệm policy, value, replay buffer và MCTS cơ bản.

## 1. Bài toán đang học

State quan sát gồm tọa độ cue/object ball. Agent không chọn góc cue tuyệt đối; action normalized 2-D được map thành:

- cue speed `V0`;
- cut angle tương đối khi aim vào object ball.

Một shot thành công nếu cue chạm object và tổng linear-cushion contacts của hai ball đúng bằng 3. Episode có 10 shots nên binary episode score nằm trong `[0,10]`.

Điểm quan trọng: `pt.aim.at_ball(...)` đã hỗ trợ hướng aim. Vì vậy trước khi gọi task “rất sparse”, phải đo random-policy contact/success rate bằng script diagnostics.

## 2. Baseline Sampled EfficientZero

Sampled EfficientZero học một latent dynamics model và dùng MCTS. Với continuous action space, mỗi node chỉ search một tập action được sample. Study này khóa:

- K=20 sampled actions;
- 50 simulations;
- replay ratio=1;
- reanalyze=.25;
- SSL weight=2;
- learning rate=.003.

Ta không tune các tham số này vì mục tiêu là đo curriculum/reward, không phải thắng bằng compute/hyperparameter khác.

## 3. Tại sao reward densification có thể giúp

Binary reward chỉ phân biệt exact-three và mọi failure. `event_aligned` thêm progress signal nhỏ cho contact + 0/1/2 cushions, nhưng exact-three vẫn nhận 1.0.

Rủi ro: policy có thể học cách lấy reward nhỏ dễ dàng thay vì exact-three. Do đó:

- collector có thể train bằng `event_aligned`;
- evaluator luôn dùng `binary`;
- shaped return tăng nhưng binary score đứng yên là failure.

Không gọi reward này là potential-based hoặc policy-invariant.

## 4. Curriculum đang làm gì

- C0 canonical: layout gốc.
- C1 local: perturb nhẹ quanh layout gốc.
- C2 broad: random valid positions toàn table.

Schedule cố định C0 -> C1 -> C2. Đây là progressive domain expansion, không phải adaptive curriculum theo competence. Fixed schedule dễ reproduce hơn và phù hợp một GPU nhỏ.

## 5. Vì sao cần factorial A–D

Nếu chỉ so A với D và D thắng, ta không biết thắng do reward, curriculum hay interaction. Vì vậy:

| Arm | Reward | Starts |
| --- | --- | --- |
| A | sparse | fixed |
| B | dense | fixed |
| C | sparse | curriculum |
| D | dense | curriculum |

E dùng dense + broad ngay từ đầu. Nếu E ngang hoặc hơn D, điều hữu ích có thể là data diversity chứ không phải thứ tự curriculum.

## 6. Quy trình một run

### Trước run

1. Chọn arm/seed/budget từ `configs/experiment_matrix.json`.
2. Copy `templates/run_manifest.json` và `templates/experiment_entry.md` vào `records/<run-id>/`.
3. Ghi commit, dirty state, exact command, hardware, hypothesis, primary metric và stopping rule.
4. Không chỉnh code/config sau khi đã đăng ký mà không ghi `plan_deviations`.

### Trong run

- Giữ stdout/stderr.
- Theo dõi TensorBoard evaluator binary reward, loss, throughput và NaN.
- Không kết luận từ vài nghìn step đầu.
- Run crash vẫn là artifact cần ghi.

### Sau run

1. Ghi status, end time, checkpoint/log path và checksum.
2. Export TensorBoard scalars.
3. Evaluate checkpoint trên T0/T1/T2 bằng binary reward.
4. Điền result summary.
5. Chỉ sau đó mở notebook để so sánh.

## 7. Cách đọc learning curve

- Trục x phải là environment interactions, không phải chỉ wall time hoặc train iteration.
- Đường primary là canonical binary evaluation return.
- AUC đo agent học sớm hay muộn trong cùng budget.
- Final score đo quality ở cuối/selected checkpoint.
- Vẽ từng seed; mean curve không được che giấu một seed collapse.
- Không smooth quá mạnh. Luôn ghi smoothing rule.

## 8. Held-out evaluation

- **T0 canonical:** task gốc, headline metric.
- **T1 local:** robustness quanh layout gốc.
- **T2 broad:** generalization khó hơn.

Không chọn checkpoint bằng T1/T2 final test. Nếu D tăng T2 nhưng giảm mạnh T0, đó là trade-off chứ không tự động là thắng.

## 9. Statistics thực dụng

Với 3 seeds:

- công bố từng seed;
- mean ± std;
- bootstrap CI cho difference A-vs-D;
- effect size theo số successful shots/episode.

Với 5 seeds A/D, có thể viết headline conclusion mạnh hơn. CI rộng nghĩa là evidence chưa chắc, không phải phải chạy đến khi ra số đẹp.

## 10. Checklist trước Vast.ai

- [ ] Repo/commit đã push hoặc archive.
- [ ] Manifest/command/seed list đã chuẩn bị.
- [ ] Test chạy local/WSL.
- [ ] Biết output directory cho từng arm/seed.
- [ ] Disk đủ cho logs/checkpoints.
- [ ] Có lệnh `scp`/download artifacts.
- [ ] Đã đo 2k-step throughput để ước lượng tiền.
- [ ] Không chạy nhiều arm với cùng output path.

## 11. Checklist trước khi destroy instance

- [ ] Download TensorBoard logs.
- [ ] Download best/final checkpoints.
- [ ] Download stdout/stderr và resolved config.
- [ ] Ghi GPU/CPU/Python/PyTorch/CUDA/package versions.
- [ ] Tạo checksum.
- [ ] Cập nhật journal status.
- [ ] Mở thử artifact sau khi download.

## 12. Các kết quả đều có giá trị

- D > A rõ: tiếp tục confirmation seeds.
- B tốt, C không: reward signal quan trọng hơn curriculum.
- C tốt, B không: state diversity/order quan trọng hơn shaping.
- E >= D: broad random exposure đủ tốt.
- Không arm nào hơn random/baseline: có thể task/config/metric sai hoặc intervention không hữu ích.
- A saturation: task canonical quá dễ để làm claim sparse reward.

Negative result tốt hơn claim sai nếu protocol sạch và artifact đầy đủ.

## 13. Tool map

- Tracker: [`progress_tracker.html`](progress_tracker.html).
- Notebook: [`notebooks/experiment_lab.ipynb`](notebooks/experiment_lab.ipynb).
- Commands/configs: [`configs/README.md`](configs/README.md).
- Journal practice: [`JOURNAL_GUIDE.md`](JOURNAL_GUIDE.md).
- Future-agent handoff: [`AGENT_CONTEXT.md`](AGENT_CONTEXT.md).
