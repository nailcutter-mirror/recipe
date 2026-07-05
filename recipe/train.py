"""
Canonical training loop for the Ralph launch track.

This file is part of the recipe — miners may patch it (subject to the
restricted-files contract). The proof-test runner invokes this script with a
fixed config; the training is deterministic given (config, seed, manifest).

Outputs written to `--out-dir`:
  checkpoint.pt         the final model state_dict
  training_log.jsonl    one JSON line per step (loss, lr, throughput, gradnorm)
  final_state.json      run summary (steps, final loss, wall-clock, total tokens)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import TokenShardDataset
from model import RalphBase, RalphConfig


@dataclass
class TrainConfig:
    # Model
    vocab_size: int = 50257
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    head_dim: int = 64
    ffn_mult: float = 8 / 3
    max_seq_len: int = 1024

    # Training
    seq_len: int = 256
    batch_size: int = 16
    micro_batch_size: int = 16  # gradient accumulation = batch_size / micro_batch_size
    total_steps: int = 200
    warmup_steps: int = 20
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # LR schedule. "cosine" = warmup then cosine decay to min_lr (legacy default).
    # "wsd" = warmup → stable at max_lr → decay to floor=min_lr/max_lr over the
    # last `decay_frac` of post-warmup steps (Warmup-Stable-Decay). `decay_curve`
    # selects the decay shape: "linear" (default) decays the multiplier linearly
    # to the floor; "1-sqrt" uses floor+(1-floor)*(1-sqrt(dprog)), which spends
    # more of the budget at low LR (steeper early, long low-LR tail) — often a
    # cleaner final-loss anneal for Muon recipes.
    schedule: str = "cosine"
    stable_frac: float = 0.8    # informational; decay_frac is authoritative
    decay_frac: float = 0.2     # fraction of post-warmup steps spent decaying
    decay_curve: str = "linear"  # "linear" | "1-sqrt"

    # Separate AdamW LR for the (tied) token-embedding / unembedding matrix. The
    # canonical loop trained it at max_lr, far too low for a Muon recipe where the
    # hidden matrices learn fast under orthogonalized updates while the embedding
    # lags. None / <=0 => fall back to max_lr (legacy behaviour).
    embed_lr: float | None = None
    embed_optimizer: str = "adamw"  # accepted for config fidelity (AdamW path)

    # Optimizer. "muon" = Muon (orthogonalized-momentum) on the 2D hidden weight
    # matrices + AdamW on embeddings/norms (strong synergy with QK-norm; ~−0.13
    # val_bpb vs AdamW at the h100_proxy scale). "adamw" = AdamW on everything.
    optimizer: str = "muon"
    muon_lr: float = 0.04
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    # Decoupled (AdamW-style) weight decay on the Muon 2D hidden matrices. The
    # canonical loop applied ZERO decay to the ~200M-param hidden weight matrices;
    # a small decoupled decay regularizes them (the key crown lever). Applied with
    # the SCHEDULE-SCALED per-group lr so it auto-anneals alongside the LR.
    muon_weight_decay: float = 0.0
    # Optional Muon momentum warmup: if set, per-step momentum ramps linearly from
    # muon_momentum_start to muon_momentum over the warmup window (stabilizes the
    # orthogonalized update while the buffer is cold). None => constant momentum.
    muon_momentum_start: float | None = None

    # Data + reproducibility
    manifest_path: str = "data/data_manifest.json"
    data_base_dir: str = "data"
    data_seed: int = 1337
    init_seed: int = 1337

    # Precision
    use_bf16: bool = True  # bf16 autocast on CUDA; ignored on CPU
    compile: bool = False  # torch.compile(mode="max-autotune"); state_dict saved from the UNCOMPILED module (op4-safe, no _orig_mod prefix)

    # Logging
    log_every: int = 10

    @property
    def grad_accum_steps(self) -> int:
        assert self.batch_size % self.micro_batch_size == 0
        return self.batch_size // self.micro_batch_size


def set_determinism(seed: int) -> None:
    """Set all the knobs we can to get deterministic training. Not bit-perfect
    on GPU — see whitepaper §5.2 note on cuBLAS/atomic-reduction non-determinism.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def schedule_frac(step: int, cfg: TrainConfig) -> float:
    """LR multiplier in [floor, 1.0] applied to every optimizer group's base_lr,
    where floor = min_lr / max_lr. Supports "cosine" (legacy) and "wsd".

    Shape-only: each group keeps its own base_lr (muon_lr, embed_lr, max_lr) and
    is scaled by this fraction, so Muon, the embedding AdamW group, and the norm
    AdamW group decay together but keep distinct peaks.
    """
    floor = (cfg.min_lr / cfg.max_lr) if cfg.max_lr > 0 else 0.0
    if step < cfg.warmup_steps:
        return (step + 1) / max(1, cfg.warmup_steps)

    post = step - cfg.warmup_steps
    total_post = max(1, cfg.total_steps - cfg.warmup_steps)

    if cfg.schedule == "wsd":
        decay_steps = max(1, int(round(cfg.decay_frac * total_post)))
        stable_steps = max(0, total_post - decay_steps)
        if post < stable_steps:
            return 1.0
        dprog = min(1.0, max(0.0, (post - stable_steps) / max(1, decay_steps)))
        if cfg.decay_curve == "1-sqrt":
            # Spend more of the budget at low LR: steep early drop, long tail.
            return floor + (1.0 - floor) * (1.0 - math.sqrt(dprog))
        return floor + (1.0 - floor) * (1.0 - dprog)  # linear decay to floor

    # cosine (default / legacy)
    progress = min(1.0, max(0.0, post / total_post))
    return floor + 0.5 * (1.0 - floor) * (1 + math.cos(math.pi * progress))


def cosine_lr(step: int, cfg: TrainConfig) -> float:
    """Back-compat absolute-LR helper (legacy callers / tests). Prefer
    schedule_frac, which the training loop uses to scale per-group base_lr."""
    return cfg.max_lr * schedule_frac(step, cfg)


def build_model(cfg: TrainConfig) -> RalphBase:
    return RalphBase(RalphConfig(
        vocab_size=cfg.vocab_size,
        dim=cfg.dim,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        head_dim=cfg.head_dim,
        ffn_mult=cfg.ffn_mult,
        max_seq_len=cfg.max_seq_len,
    ))


def _zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Newton-Schulz iteration to orthogonalize the update matrix (Muon).
    Computes G (G^T G)^(-1/2) approximately via a quintic iteration. Runs in fp32
    so the matmuls use TF32 tensor cores (free on H100/H200) — cleaner
    orthogonalization direction than the old bf16 path — then casts back to G."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    transpose = G.size(0) > G.size(1)
    if transpose:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transpose:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Momentum orthogonalized by Newton-Schulz, for 2D hidden weight matrices.
    See Keller Jordan's modded-nanogpt. Embeddings/heads/norms use AdamW instead."""

    def __init__(self, params, lr=0.04, momentum=0.95, nesterov=True, ns_steps=5,
                 weight_decay=0.0, momentum_start=None, warmup_steps=0):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      ns_steps=ns_steps, weight_decay=weight_decay))
        # Momentum-warmup schedule state. The training loop updates cur_step each
        # step (before opt.step()) so momentum can ramp momentum_start->momentum
        # over warmup_steps. Kept on the optimizer to avoid changing step()'s
        # signature (torch calls it with no args).
        self.momentum_start = momentum_start
        self.warmup_steps = int(warmup_steps)
        self.cur_step = 0

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            # Per-step momentum warmup: lerp(start, target, min(1, step/warmup)).
            if self.momentum_start is not None and self.warmup_steps > 0:
                frac = min(1.0, self.cur_step / self.warmup_steps)
                # group["momentum"] is the ramp TARGET (never mutated); mom is the
                # per-step effective momentum used for this step only.
                mom = self.momentum_start + (group["momentum"] - self.momentum_start) * frac
            else:
                mom = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad)
                buf = state["momentum_buffer"]
                buf.mul_(mom).add_(p.grad)
                upd = p.grad.add(buf, alpha=mom) if group["nesterov"] else buf
                upd = _zeropower_via_newtonschulz5(upd, steps=group["ns_steps"])
                # Scale so the RMS update magnitude is ~LR-invariant to matrix shape.
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                # Decoupled weight decay BEFORE the update, using the SCHEDULE-SCALED
                # per-group lr (group["lr"] is already annealed each step by the loop),
                # so the decay auto-anneals with the LR — same shape scaling as the
                # update keeps decay and update RMS-consistent per matrix.
                if wd != 0.0:
                    p.mul_(1.0 - lr * scale * wd)
                p.add_(upd, alpha=-lr * scale)


def build_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> list[torch.optim.Optimizer]:
    """Returns a LIST of optimizers stepped together. Each param group carries a
    "base_lr" that the training loop multiplies by the (warmup+schedule) fraction,
    so Muon and AdamW groups keep distinct base learning rates."""
    # Resolve the embedding/unembedding LR: honor cfg.embed_lr when set, otherwise
    # fall back to max_lr (legacy behaviour).
    embed_lr = cfg.embed_lr if (cfg.embed_lr is not None and cfg.embed_lr > 0) else cfg.max_lr

    if cfg.optimizer == "muon":
        muon_params, embed_params, norm_params = [], [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "tok_embed" in n or "lm_head" in n or "value_embed" in n:
                embed_params.append(p)  # recipe-v5: VE table trains with AdamW like the token embedding
            elif "ve_lambda" in n:
                norm_params.append(p)   # recipe-v5: VE mixing scalars — AdamW, no weight decay
            elif p.dim() >= 2:
                muon_params.append(p)
            else:
                norm_params.append(p)
        muon = Muon(
            muon_params,
            lr=cfg.muon_lr,
            momentum=cfg.muon_momentum,
            ns_steps=cfg.muon_ns_steps,
            weight_decay=cfg.muon_weight_decay,
            momentum_start=cfg.muon_momentum_start,
            warmup_steps=cfg.warmup_steps,
        )
        adamw = torch.optim.AdamW(
            [
                {"params": embed_params, "weight_decay": cfg.weight_decay, "lr": embed_lr},
                {"params": norm_params, "weight_decay": 0.0, "lr": cfg.max_lr},
            ],
            lr=cfg.max_lr,
            betas=(cfg.beta1, cfg.beta2),
        )
        for grp in muon.param_groups:
            grp["base_lr"] = cfg.muon_lr
        for grp in adamw.param_groups:
            grp["base_lr"] = grp["lr"]  # per-group peak (embed_lr vs max_lr)
        return [muon, adamw]

    # Pure-AdamW path: separate the (tied) embedding so it can take embed_lr too.
    embed_params, decay_params, no_decay_params = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "tok_embed" in n or "lm_head" in n:
            embed_params.append(p)
        elif p.dim() >= 2:
            decay_params.append(p)
        else:
            no_decay_params.append(p)
    adamw = torch.optim.AdamW(
        [
            {"params": embed_params, "weight_decay": cfg.weight_decay, "lr": embed_lr},
            {"params": decay_params, "weight_decay": cfg.weight_decay, "lr": cfg.max_lr},
            {"params": no_decay_params, "weight_decay": 0.0, "lr": cfg.max_lr},
        ],
        lr=cfg.max_lr,
        betas=(cfg.beta1, cfg.beta2),
    )
    for grp in adamw.param_groups:
        grp["base_lr"] = grp["lr"]
    return [adamw]


def _init_wandb(cfg: TrainConfig, out_dir: Path, use_wandb: bool) -> object | None:
    if not use_wandb:
        return None
    try:
        import wandb
        miner_gh = os.environ.get("RALPH_MINER_GH", "")
        miner_wallet = os.environ.get("BT_WALLET", "")
        run_config = {k: v for k, v in asdict(cfg).items()}
        if miner_gh:
            run_config["miner_github"] = miner_gh
        if miner_wallet:
            run_config["miner_wallet"] = miner_wallet
        tags = ["proof-test", f"{cfg.dim}d", f"{cfg.n_layers}L"]
        if miner_gh:
            tags.append(f"gh:{miner_gh}")
        if miner_wallet:
            tags.append(f"wallet:{miner_wallet}")
        name_prefix = f"{miner_gh}-" if miner_gh else ""
        run = wandb.init(
            entity=os.environ.get("WANDB_ENTITY", "ralphlabs-hub"),
            project=os.environ.get("WANDB_PROJECT", "ralph"),
            name=f"{name_prefix}train-{cfg.dim}d-{cfg.n_layers}L-{cfg.total_steps}s",
            config=run_config,
            dir=str(out_dir),
            tags=tags,
        )
        return run
    except Exception as e:
        print(f"[train] wandb init failed ({e}), continuing without it")
        return None


def train(cfg: TrainConfig, out_dir: Path, use_wandb: bool = False) -> dict:
    set_determinism(cfg.init_seed)
    # Enable TF32 tensor-core matmuls (free on H100/H200). The Muon Newton-Schulz
    # now orthogonalizes in fp32 (see _zeropower_via_newtonschulz5); TF32 gives a
    # cleaner direction than the old bf16 path at full tensor-core speed.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg).to(device)
    optimizers = build_optimizer(model, cfg)
    # torch.compile(mode="max-autotune") on the forward. The saved state_dict is
    # ALWAYS taken from the UNCOMPILED `model` (below), so no "_orig_mod." prefix
    # leaks into the checkpoint (op4 strict-load safe). Gated on cfg.compile and
    # overridable via RALPH_NO_COMPILE=1 (e.g. for a CPU/debug run).
    _compile = getattr(cfg, "compile", False) and os.environ.get("RALPH_NO_COMPILE") != "1"
    fwd = torch.compile(model) if _compile else model
    ds = TokenShardDataset(cfg.manifest_path, cfg.data_base_dir, cfg.seq_len, cfg.data_seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training_log.jsonl"
    log_f = log_path.open("w")

    wb_run = _init_wandb(cfg, out_dir, use_wandb)

    use_amp = cfg.use_bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    # bf16 has enough dynamic range that no GradScaler is needed (Muon orthogonalizes
    # in bf16 internally; AdamW groups are range-safe), so we step optimizers directly.

    n_params = model.num_parameters()
    n_params_no_embed = model.num_parameters(exclude_embeddings=True)
    print(f"[train] device={device} params={n_params:,} (no embeddings: {n_params_no_embed:,})")
    print(f"[train] precision={'bf16' if use_amp else 'fp32'}")
    print(f"[train] manifest tokens={ds.total_tokens:,} hash={ds.manifest.manifest_hash()[:16]}…")
    print(f"[train] steps={cfg.total_steps} batch={cfg.batch_size} micro={cfg.micro_batch_size} seq={cfg.seq_len}")
    if wb_run:
        print(f"[train] wandb: {wb_run.url}")

    start = time.time()
    tokens_seen = 0
    last_loss = float("nan")
    for step in range(cfg.total_steps):
        lr_frac = schedule_frac(step, cfg)
        lr = cfg.max_lr * lr_frac  # representative LR for logging
        # Scale each optimizer's per-group base_lr by the schedule fraction so
        # the Muon and AdamW (embedding / norm) groups keep distinct peak LRs.
        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * lr_frac
            opt.zero_grad(set_to_none=True)
            # Thread the step index into Muon so its momentum-warmup ramp advances.
            if isinstance(opt, Muon):
                opt.cur_step = step

        step_loss = 0.0
        for accum in range(cfg.grad_accum_steps):
            sub_step = step * cfg.grad_accum_steps + accum
            inp, tgt = ds.get_batch(sub_step, cfg.micro_batch_size)
            inp = inp.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                _, loss = fwd(inp, targets=tgt)
            scaled_loss = loss / cfg.grad_accum_steps
            scaled_loss.backward()
            step_loss += loss.item() / cfg.grad_accum_steps
            tokens_seen += cfg.micro_batch_size * cfg.seq_len

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
        for opt in optimizers:
            opt.step()

        last_loss = step_loss
        elapsed = time.time() - start
        tok_per_s = tokens_seen / max(elapsed, 1e-6)

        entry = {
            "step": step,
            "loss": step_loss,
            "lr": lr,
            "grad_norm": grad_norm,
            "tokens_seen": tokens_seen,
            "tokens_per_sec": tok_per_s,
            "elapsed_s": elapsed,
        }
        # recipe-v4: gate the JSONL write under log_every so long runs don't make
        # one line per step (the proof-test turns each ~10 lines into a per-epoch
        # NRAS attestation -> thousands of calls -> NRAS rate-limit/timeout).
        if step % cfg.log_every == 0 or step == cfg.total_steps - 1:
            log_f.write(json.dumps(entry) + "\n")
        log_f.flush()
        if wb_run:
            wb_run.log(entry, step=step)
        if step % cfg.log_every == 0 or step == cfg.total_steps - 1:
            print(
                f"[step {step:4d}/{cfg.total_steps}] loss={step_loss:.4f} lr={lr:.2e} "
                f"|g|={grad_norm:.2f} tok/s={tok_per_s:,.0f}",
                flush=True,
            )
        if (step % 500 == 0 and step > 0) or step == cfg.total_steps - 1:
            _ckpt_dir = out_dir / "checkpoints"
            _ckpt_dir.mkdir(exist_ok=True)
            torch.save({"model": model.state_dict(), "config": asdict(cfg), "step": step}, _ckpt_dir / f"step_{step:06d}.pt")
            with (out_dir / "progress.tsv").open("a") as _pf:
                _pf.write(f"{step}\t{step_loss:.6f}\n")
                _pf.flush()
    log_f.close()
    wb_url = None
    if wb_run:
        wb_url = wb_run.url
        try:
            history = wb_run.history(pandas=False)
            (out_dir / "wandb_metrics.json").write_text(json.dumps(history, indent=2))
            (out_dir / "wandb_run_url.txt").write_text(wb_url + "\n")
            print(f"[train] wandb metrics exported ({len(history)} steps)")
        except Exception as e:
            print(f"[train] wandb export failed ({e}), continuing")
        wb_run.finish()

    ckpt_path = out_dir / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "config": asdict(cfg)}, ckpt_path)

    summary = {
        "steps": cfg.total_steps,
        "final_loss": last_loss,
        "tokens_seen": tokens_seen,
        "wall_clock_s": time.time() - start,
        "n_params": n_params,
        "n_params_no_embed": n_params_no_embed,
        "manifest_hash": ds.manifest.manifest_hash(),
        "device": str(device),
        "precision": "bf16" if use_amp else "fp32",
        "wandb_url": wb_url,
        "config": asdict(cfg),
    }
    (out_dir / "final_state.json").write_text(json.dumps(summary, indent=2))
    print(f"[train] done. final loss={last_loss:.4f} wall={summary['wall_clock_s']:.1f}s")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=None, help="Optional JSON config override.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--total-steps", type=int, default=None)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--data-base-dir", type=Path, default=None,
                   help="Pin shard-resolution dir (runner-supplied; overrides config data_base_dir).")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--wandb", action="store_true", help="Log to Weights & Biases (requires `pip install wandb`)")
    args = p.parse_args()

    cfg = TrainConfig()
    if args.config and args.config.exists():
        overrides = json.loads(args.config.read_text())
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    if args.total_steps is not None:
        cfg.total_steps = args.total_steps
    if args.manifest is not None:
        cfg.manifest_path = str(args.manifest)
    if args.data_base_dir is not None:
        cfg.data_base_dir = str(args.data_base_dir)
    if args.seed is not None:
        cfg.init_seed = args.seed
        cfg.data_seed = args.seed

    train(cfg, args.out_dir, use_wandb=args.wandb)


if __name__ == "__main__":
    main()
