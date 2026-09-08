# pre_processing_upgrade_7 — UP-VCM

**Tiền xử lý video phổ quát cho VCM (Video Coding for Machines): một bộ
pre-filter nhỏ học được, đặt trước codec chuẩn *đóng băng* (x264 / x265),
không phụ thuộc analyzer.**

Chỉ mạng preprocessor (~44k tham số) được huấn luyện. Codec và mọi analyzer
phía sau giữ nguyên, đúng chuẩn — không sửa bitstream, không sửa decoder,
không bản đồ QP. Bộ tiền xử lý sửa pixel *trước khi encode* sao cho ở cùng
bitrate, máy vẫn nhìn thấy thứ nó cần, và phần nền tốn ít bit hơn.

```
                     (được train)               (ĐÓNG BĂNG)         (ĐÓNG BĂNG khi eval)
 video x ──► UP-VCM θ ──► x_pre ──► x264 / x265 ──► x̂ ──► Analyzer held-out ──► accuracy
             S + M1 + M2 + M3      preset medium          r2plus1d_18 (không dùng khi train)

Đại lượng tối ưu: BD-Rate(prep+codec vs codec) trên trục accuracy —
âm nghĩa là cùng độ chính xác với ít bit hơn.
```

## Kiến trúc UP-VCM (docs/MODEL_UPVCM.md)

| Thành phần | Cơ chế |
|---|---|
| **S** — importance head | CNN nhỏ (input RGB + \|Δ thời gian\|) dự đoán bản đồ importance `W ∈ [0,1]`. Lúc train được distill từ multi-teacher saliency pha DINOv2 patch energy (term `rho`); lúc deploy **không cần analyzer, không cần foundation model** |
| **M1** — background decimation | Blur mịn kênh Y + blur thô Cb/Cr (qua YCbCr, mô phỏng tổn hại yuv420), gate theo `(1−W)` — giảm entropy vùng nền |
| **M2** — ROI structure editor | UNet 2 tầng + FiLM(QP) zero-init, gate theo `W` — tăng/giữ cấu trúc vật thể ở mọi điểm rate |
| **M3** — temporal background stabilisation | Nền tĩnh được thay bằng khung *output* trước ⇒ residual liên khung ≈ 0 ⇒ motion compensation của codec tự tiết kiệm bit (module video-native) |

Cả 3 module có strength gate **zero-init** ⇒ mô hình là identity chính xác lúc
khởi tạo; mỗi cơ chế chỉ "bật" khi gradient của loss đòi hỏi.

**Loss** (`configs/upvcm_ar.yaml`):

```
L = 1.0·L_task + 3.0·L_D + 0.001·bpp + 0.05·L_temp + 10.0·L_dct + 1.0·L_W
```

- `L_task`: cross-entropy của panel teacher [r3d_18, mc3_18] (sampled từng step)
  trên reconstruction từ proxy codec yuv420 (soft→hard quantizer anneal)
- `L_W` (rho): MSE(S(x), W_target) — distill importance head
- `L_dct` (kappa): RPP adaptive-DCT rate proxy (arXiv:2301.10455) — nặn phổ
  residual thay vì bóp biên độ
- in-grid QP [30, 35, 40, 45, 50] — FiLM condition không phải ngoại suy lúc eval

## Pipeline

```
┌─ LOCAL ──────────────────────────────────────────────────────────┐
│ pytest (88 tests) → ops/smoke_local.py (train→shard eval→merge   │
│ →gates trên dữ liệu giả, không cần dataset)                      │
└──────────────────────────────────────────────────────────────────┘
┌─ KAGGLE (T4) ────────────────────────────────────────────────────┐
│ 1. probe    : fingerprint dataset + codec self-checks            │
│ 2. train    : 16 epoch, resume-aware qua session 12h             │
│              (u7-train-upvcm, repo @commit cố định)              │
│ 3. gates    : ops/gates_upvcm.py trên checkpoint (0-analyzer)    │
│              G1 non-identity / G2 no-blow-up / G3 W healthy /    │
│              G4 deploy purity                                    │
│ 4. eval     : 3 kernel shard, checkpoint gắn từ kernel output    │
│ 5. merge    : ops/merge_eval.py → BD-Rate + bootstrap CI +       │
│              gap rule → docs/RESULTS_upvcm.md                    │
└──────────────────────────────────────────────────────────────────┘
```

**Protocol đánh giá chuẩn** (khoá cố định cho mọi so sánh):
- Test set 1159 clip Kinetics-400, hash-split theo md5 clip-key
  (fingerprint `30f083f8520a`), analyzer held-out `r2plus1d_18`
- x264 **và** x265 thật (ffmpeg, preset medium), QP {30, 35, 40, 45, 50}
- Per-sequence records → merge → BD-Rate trên trục top-1 accuracy
- Bootstrap CI 95% ở mức clip (10k resamples)
- **Gap rule**: `prep − anchor ≥ −0.05` tại mọi QP trên cả hai codec —
  vi phạm là bị loại, BD đẹp cỡ nào cũng không cứu

## Chạy

```bash
# local sanity
pytest -q
python ops/smoke_local.py                       # full pipeline, dữ liệu giả

# Kaggle (KGAT pool — source ops/kaggle_env.sh <account>)
source ops/kaggle_env.sh wagur124705
python ops/push_kernel.py probe  --commit <sha> --no-gpu
python ops/push_kernel.py train  --commit <sha> --config configs/upvcm_ar.yaml \
    --accelerator NvidiaTeslaT4
python ops/gates_upvcm.py --ckpt <preprocessor.pth> --index <index.json>
python ops/push_kernel.py eval   --commit <sha> --shard-idx N --num-shards 3 \
    --train-kernel wagur124705/u7-train-upvcm --accelerator NvidiaTeslaT4
python ops/merge_eval.py <shard0> <shard1> <shard2> --out outputs/eval_upvcm_merged
```

> Lưu ý GPU: pin `--accelerator NvidiaTeslaT4` — account có thể được cấp P100
> (sm_60) không tương thích PyTorch preinstall của Kaggle.

## Cấu trúc repo

```
configs/upvcm_ar.yaml        # config chính (model + loss + protocol)
src/models/upvcm.py          # UPVCMPreprocessor (S + M1 + M2 + M3)
src/models/dino_saliency.py  # DINOv2 energy (chỉ dùng lúc train)
src/models/virtual_codec.py  # proxy codec yuv420 khả vi (block-DCT + anneal)
src/models/ste_codec.py      # cầu STE codec thật (value thật / gradient proxy)
src/engine.py                # train/eval loop, sharding, per-sequence records
src/losses.py                # composite loss (gồm rho*L_W distill)
ops/                         # Kaggle ops: push kernel, gates, merge, smoke
docs/MODEL_UPVCM.md          # thiết kế chi tiết
docs/RESULTS_upvcm.md        # kết quả (BD-Rate + CI + gap rule)
tests/                       # 88 tests (unit + integration)
```

## Tham chiếu

- DINOv2 (Oquab et al., 2023) — anchor importance lúc train
- RPP adaptive DCT penalty (arXiv:2301.10455) — hạng `kappa`
- FiLM (Perez et al., AAAI 2018) — điều kiện QP trong M2
- Kinetics-400 (Kay et al., 2017) — dữ liệu (bộ `rohanmallick/kinetics-train-5per`)
