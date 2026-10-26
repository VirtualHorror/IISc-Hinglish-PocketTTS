"""Differential learning rates with a staged unfreeze for the Hinglish fine-tune.

Three AdamW parameter groups, each on its own schedule (linear warmup from its start step, then
cosine decay to `floor_ratio` x its peak at max_steps):

  text_embedding  flow_lm.conditioner.embed (the warm-started table): starts at step 0 with the
                  highest peak lr, no weight decay (rows of pieces a batch never contains get no
                  gradient, and decay alone would shrink them, as the official optimizer notes)
  backbone        transformer, input_linear, out_norm, speaker projection, BOS embeddings: zero lr
                  until `start_step` (stage 1 trains only the table against the frozen backbone),
                  then a conservative peak
  heads           flow_net (LSD sampler head), out_eos and the objective's own parameters: as backbone

During stage 1 the frozen groups still run backward (their Adam moments warm up for free) but take
no steps, and their decoupled weight decay is lr-scaled, so they do not move at all. The official
training loop sets group["lr"] = lr_at(step) * group["lr_scale"] every step; `schedule_scales`
rewrites each group's lr_scale for the current step, relative to the reference lr it returns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

GROUPS = ("text_embedding", "backbone", "heads")
TEXT_TABLE = "flow_lm.conditioner.embed.weight"


@dataclass(frozen=True)
class GroupSchedule:
    peak_lr: float
    start_step: int = 0
    warmup_steps: int = 0


def group_of(name: str) -> str:
    if name == TEXT_TABLE:
        return "text_embedding"
    if name.startswith(("flow_lm.flow_net.", "flow_lm.out_eos.", "flow.")):
        return "heads"
    return "backbone"


def lr_at_step(step: int, sched: GroupSchedule, max_steps: int, floor_ratio: float) -> float:
    if step < sched.start_step:
        return 0.0
    s = step - sched.start_step
    if s < sched.warmup_steps:
        return sched.peak_lr * (s + 1) / sched.warmup_steps
    decay_steps = max(1, max_steps - sched.start_step - sched.warmup_steps)
    progress = min(1.0, (s - sched.warmup_steps) / decay_steps)
    floor = sched.peak_lr * floor_ratio
    return floor + 0.5 * (sched.peak_lr - floor) * (1 + math.cos(math.pi * progress))


def parse_schedules(cfg: dict[str, Any]) -> dict[str, GroupSchedule]:
    """`cfg` is the `hinglish.param_groups` section: {group: {peak_lr, start_step, warmup_steps}}."""
    missing = set(GROUPS) - set(cfg)
    if missing:
        raise ValueError(f"param_groups needs a schedule for {sorted(missing)}")
    return {g: GroupSchedule(float(cfg[g]["peak_lr"]), int(cfg[g].get("start_step", 0)),
                             int(cfg[g].get("warmup_steps", 0))) for g in GROUPS}


def build_param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    params: dict[str, list[torch.nn.Parameter]] = {g: [] for g in GROUPS}
    for name, p in model.named_parameters():
        if p.requires_grad:
            params[group_of(name)].append(p)
    return [{"params": params[g], "name": g, "lr_scale": 0.0,
             "weight_decay": 0.0 if g == "text_embedding" else weight_decay}
            for g in GROUPS if params[g]]


def schedule_scales(optimizer: torch.optim.Optimizer, schedules: dict[str, GroupSchedule], step: int,
                    max_steps: int, floor_ratio: float) -> tuple[float, dict[str, float]]:
    """Set every group's lr_scale for `step`. Returns (reference lr, lr per group): the official
    loop sets each group's lr to reference x lr_scale and logs the reference, which is the
    backbone's lr once it trains and the largest group lr before that."""
    lrs = {g["name"]: lr_at_step(step, schedules[g["name"]], max_steps, floor_ratio)
           for g in optimizer.param_groups}
    reference = lrs.get("backbone") or max(lrs.values(), default=0.0) or 1.0
    for group in optimizer.param_groups:
        group["lr_scale"] = lrs[group["name"]] / reference
    return reference, lrs
