# Evaluation protocol

[Back to README](../README.md)


Local measurements on **64 clips / 6,400 frames** at **1920 × 1080**. All methods
are scored from lossless 8-bit alpha PNGs using the MatAnyone2 metric definitions.
Average frames within each clip, then average the 64 clips with equal weight;
the first frame's dtSSD is zero and is included in the average. All metrics are
lower-is-better. MAD and MSE are multiplied by 1000, GRAD is divided by 1000,
and dtSSD is multiplied by 100 after normalizing alpha to [0, 1].

| Method | First-frame prompt | MAD ↓ | MSE ↓ | GRAD ↓ | dtSSD ↓ |
| --- | --- | ---: | ---: | ---: | ---: |
| MatAnyone2 | Segmentation mask | 1.6055 | 0.5048 | 7.1330 | 1.5301 |
| VideoMaMa | Segmentation mask | 2.4504 | 1.2844 | 4.8751 | 1.3806 |
| MAM2 (ours, checkpoint 100) | Box + clicks | **1.1270** | **0.2195** | **2.8996** | **1.1705** |

The prompt protocols differ, as indicated in the table:

- **MatAnyone2:** official released weights, supplied first-frame segmentation
  masks with 15-pixel dilation followed by erosion, 10 warmup iterations, native
  1080p inference.
- **VideoMaMa:** official released weights, with SAM2.1 Hiera L propagating the
  same initial mask; subsequent ground truth is not used as guidance. Inference
  uses consecutive 16-frame windows at 1024 × 576, seed 42, repeat-padding the
  final window and discarding padding predictions. Alpha is bilinearly restored
  to 1080p before scoring; temporal errors include window boundaries. These
  values use this local full-video inference protocol.
- **MAM2:** `checkpoint_100.pt` with the existing MEMatte weights, a first-frame
  box derived from the ground-truth alpha's fully opaque region, and saved clicks.
  These are the historical evaluator's results after correcting MEMatte RGB
  input and exporting PNG alpha. These recorded runs use historical padding.
  The current release pads each dimension independently, so a rerun may produce
  different predictions.

MAM2's per-frame metric values were checked against MatAnyone2's
`evaluation/eval_hr.py` on all 6,400 frames and matched exactly. The baseline
metric implementations were also checked against that reference. The table
reports measured checkpoint results rather than values transcribed from the paper.

## Running inference and evaluation

The release includes `mam2-benchmark infer` and `mam2-benchmark evaluate`.
Follow the [README commands](../README.md#inference--evaluation) using
[configs/youtubematte.yaml](../configs/youtubematte.yaml). Both commands read the
same dataset root and output directory. Weights are needed only for inference;
evaluation loads the saved alpha PNGs without loading either model.

The dataset contains motion/static groups with 32 clips each and 100 frames per
clip. Each clip has `har/` RGB frames and `pha/` ground-truth alpha. JPG or PNG
ground truth is supported; prediction files must be lossless grayscale 8-bit
PNGs. Numeric frame stems must match exactly, with consecutive indices. Both RGB
and alpha must have native dimensions of 1920 × 1080. Invalid or incomplete data
raises an error instead of silently dropping clips or resizing predictions.

[configs/youtubematte_prompts.yaml](../configs/youtubematte_prompts.yaml) contains
the fixed prompts from the checkpoint comparison: first-frame boxes derived
from fully opaque alpha regions, plus foreground/background clicks for eight
clips. Later-frame alpha is used only for scoring. Coordinates are in native
pixels; point labels are 1 for foreground and 0 for background. The shipped
configuration uses seed 2026, BF16 inference, and MEMatte input size 1080 × 1920.
Scoring uses float32 outside autocast.

### Mask guide

Download the
[YouTubeMatte first-frame segmentation mask archive](https://drive.google.com/file/d/1Zpa7SB7VZmkvRDiehVC-c_0dmFWXdfzK/view?usp=sharing).
The extracted `youtubematte_1920x1080/` directory contains `0000.png`–`0031.png`;
the motion and static groups use the same mask for matching clip IDs.

Use [configs/youtubematte_mask.yaml](../configs/youtubematte_mask.yaml) and set
`benchmark.first_frame_mask_root` to that directory. This selects
`benchmark.prompt_mode=mask`, replacing the box/click prompts with one binary
mask on frame zero. Masks must match the native 1920 × 1080 frame size, contain
foreground pixels, and use 8-bit values 0/255 (0/1 is also accepted). Grayscale
PNGs and RGB PNGs with identical channels are accepted. Masks are
passed directly to the predictor without dilation or erosion; no later-frame
ground truth is supplied. The default output directory is
`outputs/youtubematte_mask`.

The [README](../README.md#inference--evaluation) gives the inference command.
Evaluate the resulting PNGs with the same mask preset and output directory.
Evaluation does not open the first-frame masks, so they may be removed after
inference. Box/click and mask-guided results represent separate prompt protocols;
the historical table above reports MAM2's box/click protocol.

Set `benchmark.clips` to an explicit list to evaluate a subset. The JSON records
the evaluated clip inventory, frame stems, counts, per-frame values, per-clip
means, per-group means and overall mean. Subset results are not full-test-set
results. CPU scoring is supported via `benchmark.metrics_device=cpu`.

## Metric definitions

For normalized alpha `p` and ground truth `a`:

- **MAD:** `mean(abs(p - a)) × 1000`.
- **MSE:** `mean((p - a)²) × 1000`.
- **GRAD:** `sum((G(p) - G(a))²) / 1000`, where `G` is the magnitude of
  Gaussian derivatives with sigma 1.4, L2-normalized derivative kernels and
  replicated borders. This is a pixel sum, so scoring resolution matters.
- **dtSSD:** `sqrt(mean(((p[t] - p[t-1]) - (a[t] - a[t-1]))²)) × 100`.
  The first frame contributes zero; temporal state resets for each clip.

These formulas match
[MatAnyone2's evaluator](https://github.com/pq-yang/MatAnyone2/blob/main/evaluation/eval_hr.py).
The release implements them with PyTorch and uses strict frame pairing and
explicit clip-level averaging. For equal-length YouTubeMatte clips, this equals
the reference spreadsheet's averaging. For clips with different frame counts,
each clip still receives equal weight.

Baseline inference implementations and their checkpoints remain separate from
this release; the commands above run and evaluate MAM2.
