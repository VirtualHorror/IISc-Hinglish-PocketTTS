"""EOS prediction accuracy from teacher-forced EOS logits.

TrainableTTS trains out_eos to fire on the first frame after the target audio (the row's first
padded position) and to stay low on every valid frame. Generation (pocket_tts TTSModel) stops once
the logit exceeds `eos_threshold` (default -4.0) after at least MIN_FRAMES_BEFORE_EOS frames. The
same rule is scored here per row with a known end (rows shorter than the batch's padded length):

  eos_hit        the logit at the true end frame is above the threshold
  eos_premature  it fires on a frame before the end (generation would cut speech short)
  eos_frame_acc  per-frame accuracy over the valid frames and the end frame
  eos_err_frames mean |first firing frame - true end| in 80 ms frames (a row that never fires
                 counts as firing at the padded length)
"""

from __future__ import annotations

import torch

MIN_FRAMES_BEFORE_EOS = 6  # pocket_tts TTSModel._MIN_FRAMES_BEFORE_EOS
THRESHOLDS = (-4.0, 0.0)


def eos_statistics(logits: torch.Tensor, mask: torch.Tensor,
                   thresholds: tuple[float, ...] = THRESHOLDS) -> dict[str, float]:
    """`logits` [B, T] EOS logits, `mask` [B, T] valid target frames. Returns sums over rows with a
    known end, plus "eos_rows" and "eos_frames" so batches can be pooled before dividing."""
    logits, mask = logits.float(), mask.bool()
    B, T = mask.shape
    lengths = mask.sum(1)
    known = lengths < T
    out: dict[str, float] = {"eos_rows": float(known.sum())}
    if not known.any():
        return out
    lg, lengths = logits[known], lengths[known]
    t = torch.arange(T, device=lg.device)[None, :]
    is_end = t == lengths[:, None]
    scored = t <= lengths[:, None]  # valid frames + the end frame
    eligible = t >= MIN_FRAMES_BEFORE_EOS
    out["eos_frames"] = float(scored.sum())
    for thr in thresholds:
        fire = lg > thr
        tag = f"@{thr:g}"
        out["eos_hit" + tag] = float(fire[is_end].sum())
        out["eos_premature" + tag] = float((fire & eligible & (t < lengths[:, None])).any(1).sum())
        out["eos_frame_acc" + tag] = float(((fire == is_end) & scored).sum())
        first = torch.where(fire & eligible, t, T).min(1).values
        out["eos_err_frames" + tag] = float((first - lengths).abs().sum())
    return out


def finalize(sums: dict[str, float]) -> dict[str, float]:
    rows, frames = sums.get("eos_rows", 0.0), sums.get("eos_frames", 0.0)
    out = {}
    for k, v in sums.items():
        if k in ("eos_rows", "eos_frames"):
            continue
        denom = frames if k.startswith("eos_frame_acc") else rows
        out[k] = v / denom if denom else float("nan")
    out["eos_rows"] = rows
    return out
