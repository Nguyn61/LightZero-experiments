# Experiment entry — `<run_id>`

## Metadata

| Field | Value |
| --- | --- |
| `run_id` | `<YYYYMMDD-family-variant-seed>` |
| `status` | `planned` |
| `experiment_family` | `baseline` / `curriculum` / `reward` / `curriculum_reward` |
| `author` | `<name>` |
| `created_at` | `<ISO-8601 timestamp>` |
| `repository_commit` | `<full git SHA>` |
| `manifest` | `[run_manifest.json](run_manifest.json)` |
| `result_summary` | `[result_summary.json](result_summary.json)` |

## Trạng thái implementation

- Curriculum: **Planned** / **Implemented** / **Verified** — `<path, commit và bằng chứng hoặc lý do>`
- Reward change: **Planned** / **Implemented** / **Verified** — `<path, commit và bằng chứng hoặc lý do>`
- Baseline semantics: `<config/source đã đối chiếu hoặc chưa đối chiếu>`

> Không đổi `Planned` thành `Implemented` nếu không có commit/path; không đổi thành `Verified` nếu không có kiểm tra/artefact truy vết.

## Câu hỏi và giả thuyết

- **Question:** `<câu hỏi có thể kiểm chứng>`
- **Hypothesis:** `<giả thuyết và hướng kỳ vọng>`
- **Decision metric:** `<metric, đơn vị, aggregator>`
- **Success criterion registered before run:** `<ngưỡng/quy tắc hoặc N/A>`

## Thiết kế đã đăng ký trước

| Mục | Giá trị |
| --- | --- |
| Baseline source config | `<relative path>` |
| Treatment source config | `<relative path hoặc N/A>` |
| Chỉ một thay đổi chính | `<mô tả>` |
| Giữ cố định | `<algorithm, seed set, max_env_step, eval protocol, ...>` |
| Seeds | `<list>` |
| Budget | `<max_env_step / wall-time limit>` |
| Evaluation protocol | `<episodes, cadence, objective>` |
| Exclusion/stopping rules | `<quy tắc>` |

## Cụ thể về treatment

### Curriculum

`<N/A nếu không có>`

- Difficulty variable/stages: `<...>`
- Transition rule and cadence: `<...>`
- Train/eval/resume behavior: `<...>`
- Reproducibility and test evidence: `<...>`

### Reward

`<N/A nếu không có>`

- Baseline reward reference: `<...>`
- Treatment formula and timing: `<...>`
- Scale/clipping and logging components: `<...>`
- Evaluation objective and disable path: `<...>`

## Thực thi thực tế

- **Command executed:** `<exact command hoặc chưa chạy>`
- **Started/finished:** `<timestamp / timestamp>`
- **Runtime environment:** `<OS, Python, device, relevant dependency versions>`
- **Output/log/checkpoint paths:** `<relative paths or URIs>`
- **Deviations from plan:** `<none hoặc mô tả>`
- **Failures/retries:** `<none hoặc mô tả>`

## Kết quả

| Metric | Value | Unit | Seeds / aggregation | Evidence path |
| --- | ---: | --- | --- | --- |
| `<primary>` | `<value or null>` | `<unit>` | `<...>` | `<...>` |
| `<sample efficiency>` | `<value or null>` | `<unit>` | `<...>` | `<...>` |
| `<cost>` | `<value or null>` | `<unit>` | `<...>` | `<...>` |

## Đánh giá và bước tiếp theo

- **Comparison-ready:** `false` / `true`
- **Conclusion:** `<không xác định / ủng hộ / không ủng hộ; nêu bằng chứng>`
- **Limitations:** `<...>`
- **Next action:** `<...>`

## Checklist

- [ ] Manifest có commit, config, override, seed, command, budget và output.
- [ ] Baseline/treatment dùng cùng evaluation protocol hoặc khác biệt được nêu rõ.
- [ ] Có ghi run bị loại và lý do.
- [ ] Summary JSON hợp lệ và liên kết được từ entry này.
- [ ] Nhãn Implemented/Verified/Planned phản ánh bằng chứng thực tế.
