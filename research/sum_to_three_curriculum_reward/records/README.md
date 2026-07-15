# Hồ sơ thực nghiệm

Thư mục này lưu provenance cho từng run. **Implemented:** code/config/template nghiên cứu. **Planned:** các run và kết quả thực tế; hiện chưa có quantitative result được xác minh. Không thêm kết quả giả để làm ví dụ.

## Cấu trúc một run

```text
records/
  <run_id>/
    experiment_entry.md
    run_manifest.json
    result_summary.json
    logs/                 # tùy chọn; hoặc URI/path tới nơi lưu ngoài repo
    artifacts/            # tùy chọn; không commit tệp lớn nếu chính sách dự án không cho phép
```

Tạo `experiment_entry.md` từ [`../templates/experiment_entry.md`](../templates/experiment_entry.md), `run_manifest.json` từ [`../templates/run_manifest.json`](../templates/run_manifest.json), và `result_summary.json` từ [`../templates/result_summary.json`](../templates/result_summary.json).

## Đặt tên

Dùng dạng:

```text
YYYYMMDD-<family>-<variant>-seed<NNN>
```

Dùng arm ID từ protocol: `A_sparse_fixed`, `B_dense_fixed`, `C_sparse_curriculum`, `D_dense_curriculum` hoặc `E_dense_broad`. `<variant>` mô tả smoke/pilot/main, không mang kết luận; seed luôn được ghi rõ. Ví dụ: `20260715-D_dense_curriculum-pilot-seed000`.

Với một thí nghiệm nhiều seed, vẫn nên có hồ sơ per-seed để giữ command/output riêng. Có thể tạo entry tổng hợp riêng, nhưng nó phải liệt kê mọi `run_id` đầu vào.

## Quy tắc provenance

- Cất manifest trước lúc chạy và không sửa im lặng sau chạy; ghi thay đổi vào `plan_deviations`.
- Dùng path tương đối khi artefact nằm trong repository; dùng URI hoặc path ngoài repo khi phù hợp.
- Không commit checkpoint, video, tensorboard dump hay log lớn nếu không được yêu cầu. Lưu path/checksum/URI có thể truy vết thay thế.
- Không xóa record thất bại. Đặt `status`/`invalidated` và ghi `exclusion_reason`.
- Chỉ đặt `comparison_ready: true` sau khi kiểm tra protocol, seed set, budget và metric theo [RESEARCH_PLAN.md](../RESEARCH_PLAN.md).

Chi tiết cách viết entry: [JOURNAL_GUIDE.md](../JOURNAL_GUIDE.md).