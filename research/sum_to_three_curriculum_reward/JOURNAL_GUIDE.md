# Hướng dẫn nhật ký nghiên cứu

Nhật ký là bằng chứng có thể đọc được cho một quyết định thực nghiệm, không phải bản tóm tắt marketing. Mỗi entry dùng [mẫu `experiment_entry.md`](templates/experiment_entry.md) và liên kết tới manifest/summary của cùng `run_id`.

## Khi nào tạo hoặc cập nhật entry

| Thời điểm | Nội dung bắt buộc |
| --- | --- |
| Trước run | câu hỏi, giả thuyết, baseline/treatment, điều giữ cố định, thay đổi chính, criteria, commit và manifest path. |
| Sau run | trạng thái thực tế, command đã dùng, output/artefact, metric, lỗi/khác biệt so với kế hoạch. |
| Trước so sánh | xác nhận protocol/seed/budget tương đương, các run bị loại và phương pháp tổng hợp. |
| Sau quyết định | kết luận có mức độ bằng chứng, giới hạn, bước kế tiếp; không nâng `Planned` thành `Verified` chỉ do kỳ vọng. |

## Quy ước trạng thái

- **Planned:** giả thuyết, config hoặc run được dự định nhưng chưa có bằng chứng runtime.
- **Implemented:** thay đổi mã/tài liệu/cấu hình đã hiện diện ở một commit; ghi commit và path.
- **Verified:** có kiểm tra truy vết phù hợp. Với kết quả, tối thiểu cần manifest hoàn chỉnh, artefact/output path và đối chiếu protocol.
- **Invalidated:** run không dùng để so sánh do lỗi, lệch cấu hình, dữ liệu hỏng hoặc vi phạm protocol. Không xóa entry; nêu lý do.

`Implemented` không tự động đồng nghĩa với `Verified`; `Verified` không tự động chứng minh giả thuyết đúng.

## Cách ghi một thay đổi curriculum/reward

Khi và chỉ khi một thay đổi core đã tồn tại, entry phải nêu:

### Curriculum

- biến độ khó, stage và miền giá trị;
- stage ban đầu, điều kiện/chỉ số chuyển stage, tần suất kiểm tra;
- ảnh hưởng tới `reset`, train/eval, checkpoint/resume và seed;
- đường dẫn implementation/config, commit và kiểm thử.

### Reward

- reward baseline được đối chiếu từ đâu;
- công thức treatment, từng thành phần, scale/clipping và thời điểm phát;
- cách tắt treatment để khôi phục baseline;
- objective dùng cho evaluation và nguy cơ reward hacking;
- đường dẫn implementation/config, commit và kiểm thử.

Nếu chưa có các chi tiết này, ghi **Planned**, không điền mô tả suy đoán.

## Cách ghi số liệu

- Lưu số liệu gốc hoặc đường dẫn đến chúng; không chỉ chép một biểu đồ/số cuối.
- Ghi đơn vị (episode, env step, wall time), aggregator (mean/median) và seed set.
- Ghi `null` trong JSON khi chưa có số liệu; đừng thay bằng `0`.
- Tách metric quyết định, metric chẩn đoán và metric chi phí.
- Ghi mọi retry, run dừng sớm, NaN hoặc thay đổi dependency. Chúng có thể giải thích variance.

## Liên kết tối thiểu cho mỗi run

```text
records/<run_id>/experiment_entry.md
records/<run_id>/run_manifest.json
records/<run_id>/result_summary.json
<log/checkpoint/raw-metric path>
```

Các tên key/schema giữ bằng English để máy đọc ổn định. Phần diễn giải dành cho người đọc viết bằng tiếng Việt. Xem [records/README.md](records/README.md) và [mẫu result summary](templates/result_summary.json).

## Ví dụ observation và interpretation

**Không tốt:**

> D tốt hơn vì curriculum giúp model hiểu physics.

Câu này trộn số liệu chưa nêu với cơ chế chưa chứng minh.

**Tốt:**

> Trên T0 tại 40k steps, D đạt `x ± s` còn A đạt `y ± s` qua ba training seeds; difference bootstrap CI là `[l, u]`. Kết quả hỗ trợ hypothesis sample-efficiency ở mức exploratory. Cơ chế physics representation chưa được đo, nên chưa kết luận nguyên nhân.

## Mẫu kết luận một run

```text
Decision: supports | does_not_support | inconclusive
Primary evidence: <artifact path + metric + seed>
Protocol deviations: <none or exact list>
Alternative explanations: <variance, checkpoint selection, task ease, compute>
Next action: <one concrete experiment or stop>
```

## Từ journal sang paper

| Journal evidence | Paper section |
| --- | --- |
| frozen config, command, environment versions | Methods / Reproducibility |
| hypothesis và success criterion viết trước run | Experimental protocol |
| per-seed binary metrics | Results |
| failure/retry/exclusion | Appendix / Limitations |
| T0/T1/T2 comparison | Generalization results |
| reward-hacking check | Discussion |
| compute/wall-time log | Efficiency table |

Không copy cảm nhận trong journal thành claim. Mỗi claim trong abstract/results phải trỏ được về result summary và raw artifact.