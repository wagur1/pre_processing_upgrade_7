# RESULTS — Round (b) QPC (điền khi có kết quả)

> **Template này được tạo TRƯỚC khi chạy** theo kỷ luật pre-registration của
> lineage (band kỳ vọng đã đóng trong `docs/RUN_DESIGN_qpc.md` từ 2026-09-04).
> Không sửa band, không sửa ngưỡng gate sau khi thấy dữ liệu.

## Trạng thái chạy

| Bước | Trạng thái | Ghi chú |
|---|---|---|
| Probe hạ tầng (fingerprint, ffmpeg, deps) | ☐ | kernel `u7-probe` |
| Train 16ep Round (b) | ☐ | kernel `u7-train`, commit `8c37e4e09a18` |
| Gates 0-GPU (regime / conditionability / no-blow-up) | ☐ | `ops/gates_qpc.py` |
| Eval sharded (protocol chuẩn n=1159) | ☐ | kernels `u7-eval-shard*` |
| Merge + bootstrap CI + gap rule | ☐ | `ops/merge_eval.py` |

## Comparators (đóng từ trước — từ v6, cùng instrument)

| Biến thể | BD h264 [CI95] | BD h265 [CI95] | gap rule |
|---|---|---|---|
| kappa=10 @16ep (lineage best, seed 0) | −3.42% [−5.88, −0.88] | −2.63% [−4.49, −0.79] | PASS |
| kappa=10 @16ep (rep1, seed 1) | −2.52% [−4.92, +0.16] | −2.33% [−4.14, −0.57] | PASS |

Đọc với dung sai ±1pp (re-run noise tầng eval/codec, phát hiện rep1).

## Band kỳ vọng đã đăng ký (TRƯỚC khi chạy)

- **Center (~35%):** h264 −2.5…−3.5 / h265 −2.3…−3.2 — net wash vs kappa=10.
- **Upside (~15%):** h264 −4…−5 / h265 −3…−4.2 — edit dồn về QP nặng; signature:
  đường Δbpp@QP30 hạ về ≤+10% trong khi gap QP45/50 giữ/leo.
- **Downside (~50%):** h264 −1.5…−2.5 / h265 −1.5…−2.5 — FiLM thành amplitude
  nuisance hoặc không dùng được conditioning trong 16ep.

## Kết quả đo

### Gates (in kèm tham chiếu incumbent theo standing rule C6(c))

```
[Gate 1 regime]      added-HF @s=0.25: QP30 __%  QP50 __%  band [−5%,+30%]
[Gate 2 condition.]  HF spread __%  RMS spread __%  (twin ±3%)  → UTILIZED / NULL
[Gate 3 no-blow-up]  RMS @s=1.0 ____ (threshold 0.14, incumbent 0.1202)
[FiLM audit]         QP30 ‖γ‖=____ ‖β‖=____ | QP50 ‖γ‖=____ ‖β‖=____
```

### Điểm rơi band

☐ Center / ☐ Upside / ☐ Downside — theo định nghĩa trên.

### BD-Rate (protocol chuẩn, n=1159, held-out r2plus1d_18, QP30–50)

| Codec | BD-Rate | CI95 (bootstrap clip-level) | P(BD<0) | gap rule |
|---|---|---|---|---|
| h264 | | | | |
| h265 | | | | |

### Per-QP (dán từ merged_results.json)

| QP | anchor h264 | prep h264 | Δbpp h264 | anchor h265 | prep h265 | Δbpp h265 |
|---|---|---|---|---|---|---|
| 30 | | | | | | |
| 35 | | | | | | |
| 40 | | | | | | |
| 45 | | | | | | |
| 50 | | | | | | |

## Đọc kết quả (viết sau khi có số, theo luật lineage)

- So với comparator cùng instrument (trên), với dung sai ±1pp.
- Nếu band downside + gate 2 NULL → "conditioning available but unused at
  this budget" — KHÔNG phải "targeting falsified" (guard (i)/(ii)).
- Nếu null-with-FiLM-USED → escalation b' đã đăng ký (BD-weighted QP
  sampling) là follow-up, không chạy tự ý trong round này.
