"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.

Hard stop: `train_gpt.py` and `train_gpt_mlx.py` must never be longer than 1500 lines.
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import struct
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------
# HYPERPARAMETERS
# -----------------------------
# Default Simple Baseline run:
# - 9 transformer blocks at width 512
# - 8 attention heads with 4 KV heads (GQA) and 2x MLP expansion
# - vocab size 1024, sequence length 1024, tied embeddings
# - 524,288 train tokens per step for 20,000 iterations with a ~10 minute cap

class Hyperparameters:
    # Data paths are shard globs produced by the existing preprocessing pipeline.
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    init_model_path = os.environ.get("INIT_MODEL_PATH", "")
    init_model_strict = bool(int(os.environ.get("INIT_MODEL_STRICT", "1")))
    seed = int(os.environ.get("SEED", 1337))

    # Validation cadence and batch size. Validation always uses the full fineweb_val split.
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    validate_at_step_zero = bool(int(os.environ.get("VALIDATE_AT_STEP_ZERO", "0")))

    use_projection_qat = bool(int(os.environ.get("USE_PROJECTION_QAT", "0")))
    qat_start_ms = float(os.environ.get("QAT_START_MS", 540000.0))
    qat_lr_scale = float(os.environ.get("QAT_LR_SCALE", 0.25))
    qat_every_steps = int(os.environ.get("QAT_EVERY_STEPS", 4))

    # Training length.
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    _unique_default = os.environ.get("NUM_UNIQUE_ENGINES", "9")
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # Optimizer hyperparameters.
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

    # Factorized embeddings
    use_factor_embed = bool(int(os.environ.get("USE_FACTOR_EMBED", "0")))
    embed_dim = int(os.environ.get("EMBED_DIM", 128))

    # Partial sharing
    num_unique_attn = int(os.environ.get("NUM_UNIQUE_ATTN", _unique_default))
    num_unique_mlp = int(os.environ.get("NUM_UNIQUE_MLP", _unique_default))

    # Fast online adapters
    use_fast_adapters = bool(int(os.environ.get("USE_FAST_ADAPTERS", "0")))
    fast_rank = int(os.environ.get("FAST_RANK", 16))
    fast_grad_clip = float(os.environ.get("FAST_GRAD_CLIP", 0.1))
    fast_gate_init = float(os.environ.get("FAST_GATE_INIT", 0.05))

    # Meta-style training schedule
    train_meta_every = int(os.environ.get("TRAIN_META_EVERY", 2))
    train_meta_start_frac = float(os.environ.get("TRAIN_META_START_FRAC", 0.70))
    train_meta_end_frac = float(os.environ.get("TRAIN_META_END_FRAC", 0.90))
    train_meta_steps = int(os.environ.get("TRAIN_META_STEPS", 1))
    train_meta_first_order = bool(int(os.environ.get("TRAIN_META_FIRST_ORDER", 1)))
    train_meta_min_seqs_per_side = int(os.environ.get("TRAIN_META_MIN_SEQS_PER_SIDE", 4))
    train_meta_aux_weight = float(os.environ.get("TRAIN_META_AUX_WEIGHT", 0.25))
    train_meta_loss_a_weight = float(os.environ.get("TRAIN_META_LOSS_A_WEIGHT", 0.2))
    train_meta_loss_b_weight = float(os.environ.get("TRAIN_META_LOSS_B_WEIGHT", 0.8))
    train_meta_adapter_only = bool(int(os.environ.get("TRAIN_META_ADAPTER_ONLY", 1)))
    train_meta_freeze_backbone = bool(int(os.environ.get("TRAIN_META_FREEZE_BACKBONE", 0)))
    train_meta_same_shard = bool(int(os.environ.get("TRAIN_META_SAME_SHARD", 1)))

    # Streaming eval with causal adaptation
    eval_window = int(os.environ.get("EVAL_WINDOW", 4096))
    eval_adapt = int(os.environ.get("EVAL_ADAPT", 3072))
    eval_score = int(os.environ.get("EVAL_SCORE", 1024))
    eval_ttt_steps = int(os.environ.get("EVAL_TTT_STEPS", 0))

    # Compile behavior
    use_compile = bool(int(os.environ.get("USE_COMPILE", "1")))
    compile_fullgraph = bool(int(os.environ.get("COMPILE_FULLGRAPH", "0")))

    def __init__(self):
        # Data paths are shard globs produced by the existing preprocessing pipeline.
        self.data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
        self.train_files = os.path.join(self.data_path, "fineweb_train_*.bin")
        self.val_files = os.path.join(self.data_path, "fineweb_val_*.bin")
        self.tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
        self.run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
        self.init_model_path = os.environ.get("INIT_MODEL_PATH", "")
        self.init_model_strict = bool(int(os.environ.get("INIT_MODEL_STRICT", "1")))
        self.seed = int(os.environ.get("SEED", 1337))

        # Validation cadence and batch size. Validation always uses the full fineweb_val split.
        self.val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
        self.train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))
        self.validate_at_step_zero = bool(int(os.environ.get("VALIDATE_AT_STEP_ZERO", "0")))

        self.use_projection_qat = bool(int(os.environ.get("USE_PROJECTION_QAT", "0")))
        self.qat_start_ms = float(os.environ.get("QAT_START_MS", 540000.0))
        self.qat_lr_scale = float(os.environ.get("QAT_LR_SCALE", 0.25))
        self.qat_every_steps = int(os.environ.get("QAT_EVERY_STEPS", 4))

        # Training length.
        self.iterations = int(os.environ.get("ITERATIONS", 20000))
        self.warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
        self.warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
        self.train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
        self.train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
        self.max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
        self.qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

        # Model shape.
        self.vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
        self.num_layers = int(os.environ.get("NUM_LAYERS", 9))
        self.num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
        self.model_dim = int(os.environ.get("MODEL_DIM", 512))
        unique_default = os.environ.get("NUM_UNIQUE_ENGINES", "9")
        self.num_heads = int(os.environ.get("NUM_HEADS", 8))
        self.mlp_mult = int(os.environ.get("MLP_MULT", 2))
        self.tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
        self.rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
        self.logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

        # Optimizer hyperparameters.
        self.embed_lr = float(os.environ.get("EMBED_LR", 0.6))
        self.head_lr = float(os.environ.get("HEAD_LR", 0.008))
        self.tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
        self.tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
        self.matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
        self.scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
        self.muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
        self.muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
        self.muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
        self.muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
        self.beta1 = float(os.environ.get("BETA1", 0.9))
        self.beta2 = float(os.environ.get("BETA2", 0.95))
        self.adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
        self.grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

        # Factorized embeddings
        self.use_factor_embed = bool(int(os.environ.get("USE_FACTOR_EMBED", "0")))
        self.embed_dim = int(os.environ.get("EMBED_DIM", 128))

        # Partial sharing
        self.num_unique_attn = int(os.environ.get("NUM_UNIQUE_ATTN", unique_default))
        self.num_unique_mlp = int(os.environ.get("NUM_UNIQUE_MLP", unique_default))

        # Fast online adapters
        self.use_fast_adapters = bool(int(os.environ.get("USE_FAST_ADAPTERS", "0")))
        self.fast_rank = int(os.environ.get("FAST_RANK", 16))
        self.fast_grad_clip = float(os.environ.get("FAST_GRAD_CLIP", 0.1))
        self.fast_gate_init = float(os.environ.get("FAST_GATE_INIT", 0.05))

        # Meta-style training schedule
        self.train_meta_every = int(os.environ.get("TRAIN_META_EVERY", 2))
        self.train_meta_start_frac = float(os.environ.get("TRAIN_META_START_FRAC", 0.70))
        self.train_meta_end_frac = float(os.environ.get("TRAIN_META_END_FRAC", 0.90))
        self.train_meta_steps = int(os.environ.get("TRAIN_META_STEPS", 1))
        self.train_meta_first_order = bool(int(os.environ.get("TRAIN_META_FIRST_ORDER", 1)))
        self.train_meta_min_seqs_per_side = int(os.environ.get("TRAIN_META_MIN_SEQS_PER_SIDE", 4))
        self.train_meta_aux_weight = float(os.environ.get("TRAIN_META_AUX_WEIGHT", 0.25))
        self.train_meta_loss_a_weight = float(os.environ.get("TRAIN_META_LOSS_A_WEIGHT", 0.2))
        self.train_meta_loss_b_weight = float(os.environ.get("TRAIN_META_LOSS_B_WEIGHT", 0.8))
        self.train_meta_adapter_only = bool(int(os.environ.get("TRAIN_META_ADAPTER_ONLY", 1)))
        self.train_meta_freeze_backbone = bool(int(os.environ.get("TRAIN_META_FREEZE_BACKBONE", 0)))
        self.train_meta_same_shard = bool(int(os.environ.get("TRAIN_META_SAME_SHARD", 1)))

        # Streaming eval with causal adaptation
        self.eval_window = int(os.environ.get("EVAL_WINDOW", 4096))
        self.eval_adapt = int(os.environ.get("EVAL_ADAPT", 3072))
        self.eval_score = int(os.environ.get("EVAL_SCORE", 1024))
        self.eval_ttt_steps = int(os.environ.get("EVAL_TTT_STEPS", 0))

        # Compile behavior
        self.use_compile = bool(int(os.environ.get("USE_COMPILE", "1")))
        self.compile_fullgraph = bool(int(os.environ.get("COMPILE_FULLGRAPH", "0")))


CLI_OVERRIDE_ENV_KEYS = frozenset({
    "ADAM_EPS",
    "BETA1",
    "BETA2",
    "COMPILE_FULLGRAPH",
    "CONTROL_TENSOR_NAME_PATTERNS",
    "DATA_PATH",
    "EMBED_DIM",
    "EMBED_LR",
    "EVAL_ADAPT",
    "EVAL_SCORE",
    "EVAL_TTT_STEPS",
    "EVAL_WINDOW",
    "FAST_GRAD_CLIP",
    "FAST_GATE_INIT",
    "FAST_RANK",
    "GRAD_CLIP_NORM",
    "HEAD_LR",
    "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
    "INIT_MODEL_PATH",
    "INIT_MODEL_STRICT",
    "ITERATIONS",
    "LOCAL_RANK",
    "LOGIT_SOFTCAP",
    "MATRIX_LR",
    "MAX_WALLCLOCK_SECONDS",
    "MLP_MULT",
    "MODEL_DIM",
    "MUON_BACKEND_STEPS",
    "MUON_MOMENTUM",
    "MUON_MOMENTUM_WARMUP_START",
    "MUON_MOMENTUM_WARMUP_STEPS",
    "NUM_HEADS",
    "NUM_KV_HEADS",
    "NUM_LAYERS",
    "NUM_UNIQUE_ATTN",
    "NUM_UNIQUE_ENGINES",
    "NUM_UNIQUE_MLP",
    "QAT_EVERY_STEPS",
    "QAT_LR_SCALE",
    "QAT_START_MS",
    "QK_GAIN_INIT",
    "RANK",
    "ROPE_BASE",
    "RUN_ID",
    "SCALAR_LR",
    "SEED",
    "TIE_EMBEDDINGS",
    "TIED_EMBED_INIT_STD",
    "TIED_EMBED_LR",
    "TOKENIZER_PATH",
    "TRAIN_BATCH_TOKENS",
    "TRAIN_LOG_EVERY",
    "TRAIN_META_END_FRAC",
    "TRAIN_META_EVERY",
    "TRAIN_META_FIRST_ORDER",
    "TRAIN_META_AUX_WEIGHT",
    "TRAIN_META_LOSS_A_WEIGHT",
    "TRAIN_META_LOSS_B_WEIGHT",
    "TRAIN_META_ADAPTER_ONLY",
    "TRAIN_META_FREEZE_BACKBONE",
    "TRAIN_META_MIN_SEQS_PER_SIDE",
    "TRAIN_META_SAME_SHARD",
    "TRAIN_META_START_FRAC",
    "TRAIN_META_STEPS",
    "TRAIN_SEQ_LEN",
    "USE_COMPILE",
    "USE_FACTOR_EMBED",
    "USE_FAST_ADAPTERS",
    "USE_PROJECTION_QAT",
    "VALIDATE_AT_STEP_ZERO",
    "VAL_LOSS_EVERY",
    "VAL_TOKENS_LIMIT",
    "VOCAB_SIZE",
    "WARMDOWN_ITERS",
    "WARMUP_STEPS",
    "WORLD_SIZE",
})

def parse_cli_overrides(argv: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    idx = 0
    while idx < len(argv):
        token = argv[idx]
        if token in {"-h", "--help"}:
            raise SystemExit(
                "Usage: python train_gpt_moonshot.py [--name value | --name=value] ...\n"
                "CLI overrides map to env-backed settings and take precedence over environment variables.\n"
                "Examples: --qat-every-steps 1 --train-meta-first-order=0 --use-fast-adapters 1"
            )
        if not token.startswith("--"):
            raise ValueError(f"Unexpected positional argument '{token}'. Use --name value overrides.")
        if "=" in token:
            raw_key, raw_value = token[2:].split("=", 1)
        else:
            if idx + 1 >= len(argv):
                raise ValueError(f"Missing value for CLI override '{token}'")
            raw_key = token[2:]
            raw_value = argv[idx + 1]
            idx += 1
        env_key = raw_key.replace("-", "_").upper()
        if env_key not in CLI_OVERRIDE_ENV_KEYS:
            raise ValueError(
                f"Unknown CLI override '{token}'. Use env-backed names like --qat-every-steps 1 "
                "or --train-meta-first-order 0."
            )
        overrides[env_key] = raw_value
        idx += 1
    return overrides

LOW_PRECISION_DTYPE = torch.bfloat16


def resolve_low_precision_dtype(device: torch.device) -> torch.dtype:
    override = os.environ.get("LOW_PRECISION_DTYPE", "auto").strip().lower()
    if override in {"", "auto"}:
        return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    if override in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if override in {"fp16", "float16"}:
        return torch.float16
    raise ValueError(
        f"LOW_PRECISION_DTYPE must be one of auto, bf16, bfloat16, fp16, float16; got {override!r}"
    )


# -----------------------------
# MUON OPTIMIZER 
# -----------------------------
# 
# As borrowed from modded-nanogpt
# Background on Muon: https://kellerjordan.github.io/posts/muon/

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    # Muon uses this to normalize matrix-shaped gradients before applying them.
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(dtype=LOW_PRECISION_DTYPE)
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=LOW_PRECISION_DTYPE)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    # Scale correction from Muon reference implementations.
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION SETUP 
# -----------------------------
#
# It's common for small models have a large fraction of their parameters be embeddings, since the 2 * d_model * d_vocab vectors can be gigantic.
# Instead of locking the tokenizer, we let you bring your own and calculate our validation metrics on the average compression of the validation set.
# We calculate BPB (bits-per-byte) instead of validation loss, so we need methods to count the number of bits per token in the tokenizer.
# Note: Submissions that edit the tokenizer will be examined more carefully, since screwing this up might unjustly improve your score.

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    limit = int(os.environ.get("VAL_TOKENS_LIMIT", "0"))
    if limit > 0 and tokens.numel() > limit:
        tokens = tokens[:limit]
    if tokens.numel() <= 1:
        raise ValueError("Validation split is too short")
    return tokens


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    fast_state_keys: tuple[str, ...] = (),
    adapters: tuple[LowRankFastAdapter, ...] = (),
) -> tuple[float, float]:
    total_targets = val_tokens.numel() - 1
    num_blocks = math.ceil(total_targets / args.eval_score)

    block_start = (num_blocks * rank) // world_size
    block_end = (num_blocks * (rank + 1)) // world_size

    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()

    for block_id in range(block_start, block_end):
        score_start = block_id * args.eval_score
        score_end = min(score_start + args.eval_score, total_targets)

        prefix_start = max(0, score_start - args.eval_adapt)
        window_start = prefix_start
        window_end = score_end

        local = val_tokens[window_start : window_end + 1].to(device=device, dtype=torch.int64, non_blocking=True)
        x_full = local[:-1].unsqueeze(0)
        y_full = local[1:].unsqueeze(0)

        prefix_len = score_start - prefix_start
        fast_state = None
        if prefix_len > 0 and args.eval_ttt_steps > 0 and adapters:
            fast_state = init_fast_state(args.fast_rank, device, fast_state_keys)
            x_prefix = x_full[:, :prefix_len]
            y_prefix = y_full[:, :prefix_len]
            fast_state = adapt_fast_state(
                model,
                fast_state,
                x_prefix,
                y_prefix,
                steps=args.eval_ttt_steps,
                clip=args.fast_grad_clip,
                create_graph=False,
                adapters=adapters,
            )

        mask = torch.zeros_like(y_full, dtype=torch.float32)
        mask[:, prefix_len : prefix_len + (score_end - score_start)] = 1.0

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=LOW_PRECISION_DTYPE, enabled=True):
                batch_loss = model(x_full, y_full, fast_state=fast_state, loss_mask=mask)

        batch_token_count = float(score_end - score_start)
        val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
        val_token_count += batch_token_count

        prev_ids = x_full.reshape(-1)[prefix_len : prefix_len + (score_end - score_start)]
        tgt_ids = y_full.reshape(-1)[prefix_len : prefix_len + (score_end - score_start)]
        token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
        token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
        val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)

# -----------------------------
# POST-TRAINING QUANTIZATION
# -----------------------------
#
# It's silly to export our model, which is trained in bf16 and fp32, at that same precision.
# Instead, we get approximately the same model (with a small hit) by quantizing the model to int8 & zlib compressing.
# We can then decompress the model and run in higher precision for evaluation, after closing in under the size limit.

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0

def build_fast_state_keys(num_unique_attn: int, num_unique_mlp: int) -> tuple[str, ...]:
    keys = [f"attn_v_{i}" for i in range(num_unique_attn)]
    keys.extend(f"mlp_fc_{i}" for i in range(num_unique_mlp))
    keys.append("out")
    return tuple(keys)

FAST_STATE_KEYS = build_fast_state_keys(3, 3)

def get_export_project_params(model: nn.Module) -> tuple[tuple[str, Tensor], ...]:
    cached = getattr(model, "_export_project_params", None)
    if cached is None:
        cached = tuple(
            (name, p)
            for name, p in model.named_parameters()
            if p.is_floating_point()
            and p.numel() > INT8_KEEP_FLOAT_MAX_NUMEL
            and not any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS)
        )
        setattr(model, "_export_project_params", cached)
    return cached

def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())

def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t

def pack_ternary(ternary_tensor: Tensor) -> Tensor:
    shifted = ternary_tensor.flatten() + 1 
    pad_len = (5 - (shifted.numel() % 5)) % 5
    if pad_len > 0:
        shifted = torch.cat([shifted, torch.zeros(pad_len, device=shifted.device, dtype=shifted.dtype)])
    shifted = shifted.view(-1, 5).to(torch.uint8)
    powers = torch.tensor([1, 3, 9, 27, 81], dtype=torch.uint8, device=shifted.device)
    packed = (shifted * powers).sum(dim=1, dtype=torch.uint8)
    return packed.cpu()

def unpack_ternary(packed: Tensor, original_shape: tuple, device: torch.device) -> Tensor:
    powers = torch.tensor([1, 3, 9, 27, 81], dtype=torch.int32, device=device)
    unpacked = (packed.to(device).unsqueeze(1).to(torch.int32) // powers) % 3
    unpacked = unpacked.flatten() - 1
    numel = math.prod(original_shape)
    return unpacked[:numel].view(original_shape).to(LOW_PRECISION_DTYPE)

def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        # Matrices get one scale per row, which usually tracks output-channel
        # ranges much better than a single tensor-wide scale.
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    # Vectors / scalars use a simpler per-tensor scale.
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale

def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    # Single supported clean-script export format:
    # - per-row int8 for 2D float tensors
    # - per-tensor int8 for other float tensors
    # - exact passthrough for non-floats
    # - passthrough for small float tensors, stored as fp16 to save bytes
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if name.endswith(".weight_latent"):
            scale = t.abs().mean(dim=1, keepdim=True).clamp(min=1e-5).to(dtype=INT8_PER_ROW_SCALE_DTYPE)
            ternary = torch.round(t / scale).clamp(-1, 1).to(torch.int8)
            packed = pack_ternary(ternary)
            qmeta[name] = {"scheme": "ternary_packed", "shape": list(t.shape)}
            quantized[name] = packed
            scales[name] = scale
            dtypes[name] = "ternary"
            stats["int8_payload_bytes"] += tensor_nbytes(packed) + tensor_nbytes(scale)
            continue

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue

        # Small float tensors are cheap enough to keep directly. We still downcast
        # fp32/bf16 passthrough tensors to fp16 so metadata does not dominate size.
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats

def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        qmeta_item = qmeta.get(name, {})
        scheme = qmeta_item.get("scheme")
        if scheme == "ternary_packed":
            scale = obj["scales"][name].to(dtype=torch.float32).view(-1, 1)
            shape = tuple(qmeta_item["shape"])
            ternary = unpack_ternary(q, shape, "cpu")
            sparsity = ternary.float().abs().mean(dim=1, keepdim=True).clamp(min=1e-5)
            out[name] = (ternary.float() * scale / sparsity).to(LOW_PRECISION_DTYPE).contiguous()
            continue

        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            # Broadcast the saved row scale back across trailing dimensions.
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        # Restore small tensors, undoing the temporary fp16 storage cast if needed.
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


# -----------------------------
# PGQ2 CUSTOM BINARY EXPORT
# -----------------------------
DTYPE_MAP = {
    "torch.int8": 1, "torch.float16": 2, "torch.bfloat16": 3,
    "torch.float32": 4, "torch.int32": 5, "torch.int64": 6,
    "ternary": 7, "torch.uint8": 8, "torch.bool": 9,
}
INV_DTYPE_MAP = {v: k for k, v in DTYPE_MAP.items()}

def pack_pgq2(quant_obj: dict) -> bytes:
    buf = bytearray()
    buf.extend(b"PGQ2")
    buf.extend(struct.pack("<H", 1)) # version
    
    all_names = sorted(set(quant_obj["quantized"].keys()) | set(quant_obj["passthrough"].keys()))
    buf.extend(struct.pack("<H", len(all_names)))
    
    qmeta = quant_obj.get("qmeta", {})
    dtypes = quant_obj.get("dtypes", {})
    passthrough_orig_dtypes = quant_obj.get("passthrough_orig_dtypes", {})
    
    def get_dt_code(dt_str):
        if dt_str == "ternary": return 7
        if not dt_str.startswith("torch."): dt_str = "torch." + dt_str
        return DTYPE_MAP.get(dt_str, 1)
        
    for name in all_names:
        name_bytes = name.encode("utf-8")
        buf.append(len(name_bytes))
        buf.extend(name_bytes)
        
        if name in quant_obj["quantized"]:
            q = quant_obj["quantized"][name]
            s = quant_obj["scales"][name]
            scheme_str = qmeta.get(name, {}).get("scheme", "per_tensor")
            orig_dt_str = dtypes[name]
            
            if scheme_str == "ternary_packed":
                scheme_code = 12
            elif scheme_str == "per_row":
                scheme_code = 2
            else:
                scheme_code = 1
        else:
            q = quant_obj["passthrough"][name]
            s = None
            scheme_code = 10 if q.is_floating_point() else 11
            orig_dt_str = passthrough_orig_dtypes.get(name, str(q.dtype))
        
        buf.append(scheme_code)
        buf.append(get_dt_code(orig_dt_str))
        buf.append(get_dt_code(str(q.dtype)))
        
        shape = qmeta.get(name, {}).get("shape", list(q.shape))
        buf.append(len(shape))
        for dim in shape:
            buf.extend(struct.pack("<I", dim))
        
        if scheme_code in (2, 12):
            scale_bytes = s.contiguous().cpu().numpy().tobytes()
            buf.append(get_dt_code(str(s.dtype)))
            buf.extend(struct.pack("<I", len(scale_bytes)))
            buf.extend(scale_bytes)
        elif scheme_code == 1:
            buf.extend(struct.pack("<f", float(s.item())))
            
        q_bytes = q.contiguous().cpu().numpy().tobytes()
        buf.extend(struct.pack("<I", len(q_bytes)))
        buf.extend(q_bytes)
        
    return bytes(buf)

def unpack_pgq2(data: bytes) -> dict:
    obj = {
        "quantized": {}, "scales": {}, "dtypes": {}, 
        "passthrough": {}, "passthrough_orig_dtypes": {}, "qmeta": {}
    }
    offset = 4
    version, = struct.unpack_from("<H", data, offset)
    offset += 2
    num_tensors, = struct.unpack_from("<H", data, offset)
    offset += 2
    
    def read_tensor(dt_str, shape, payload_bytes):
        if dt_str == "torch.bfloat16":
            return torch.frombuffer(bytearray(payload_bytes), dtype=torch.bfloat16).clone().view(shape)
        elif dt_str in ("ternary", "torch.uint8"):
            return torch.frombuffer(bytearray(payload_bytes), dtype=torch.uint8).clone().view(shape)
        elif dt_str == "torch.bool":
            return torch.frombuffer(bytearray(payload_bytes), dtype=torch.bool).clone().view(shape)
        else:
            np_dt = getattr(np, dt_str.replace("torch.", ""))
            return torch.from_numpy(np.frombuffer(payload_bytes, dtype=np_dt).copy()).view(shape)
            
    for _ in range(num_tensors):
        name_len = data[offset]
        offset += 1
        name = data[offset:offset+name_len].decode("utf-8")
        offset += name_len
        
        scheme_code = data[offset]
        offset += 1
        orig_dtype_str = INV_DTYPE_MAP[data[offset]]
        offset += 1
        payload_dtype_str = INV_DTYPE_MAP[data[offset]]
        offset += 1
        
        ndim = data[offset]
        offset += 1
        shape = []
        for _ in range(ndim):
            dim, = struct.unpack_from("<I", data, offset)
            shape.append(dim)
            offset += 4
            
        if scheme_code in (2, 12):
            scale_dt_str = INV_DTYPE_MAP[data[offset]]
            offset += 1
            slen, = struct.unpack_from("<I", data, offset)
            offset += 4
            s = read_tensor(scale_dt_str, -1, data[offset:offset+slen])
            offset += slen
            
            qlen, = struct.unpack_from("<I", data, offset)
            offset += 4
            q_shape = (qlen,) if scheme_code == 12 else shape
            q = read_tensor(payload_dtype_str, q_shape, data[offset:offset+qlen])
            offset += qlen
            
            obj["quantized"][name] = q
            obj["scales"][name] = s
            obj["dtypes"][name] = orig_dtype_str.replace("torch.", "")
            if scheme_code == 12:
                obj["qmeta"][name] = {"scheme": "ternary_packed", "shape": shape}
            else:
                obj["qmeta"][name] = {"scheme": "per_row", "axis": 0}
                
        elif scheme_code == 1:
            scale, = struct.unpack_from("<f", data, offset)
            offset += 4
            s = torch.tensor(scale, dtype=torch.float32)
            
            qlen, = struct.unpack_from("<I", data, offset)
            offset += 4
            q = read_tensor(payload_dtype_str, shape, data[offset:offset+qlen])
            offset += qlen
            
            obj["quantized"][name] = q
            obj["scales"][name] = s
            obj["dtypes"][name] = orig_dtype_str.replace("torch.", "")
            
        else:
            qlen, = struct.unpack_from("<I", data, offset)
            offset += 4
            q = read_tensor(payload_dtype_str, shape, data[offset:offset+qlen])
            offset += qlen
            
            obj["passthrough"][name] = q
            if orig_dtype_str != payload_dtype_str:
                obj["passthrough_orig_dtypes"][name] = orig_dtype_str.replace("torch.", "")
                
    return obj


@torch.no_grad()
def project_model_to_export_grid(model: nn.Module) -> None:
    for name, p in get_export_project_params(model):
        if name.endswith(".weight_latent"):
            scale = p.abs().mean(dim=1, keepdim=True).clamp(min=1e-5)
            ternary = torch.round(p / scale).clamp(-1, 1)
            p.data.copy_((ternary * scale).to(dtype=p.dtype))
            continue
            
        q, s = quantize_float_tensor(p.data)
        if s.ndim > 0:
            restored = q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1))).float()
        else:
            restored = q.float() * float(s.item())
        p.data.copy_(restored.to(dtype=p.dtype))


# -----------------------------
# DATA LOADING 
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    # SHARD HEADER INTS & SHARD_MAGIC
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    # Reads shards sequentially and wraps around forever. The training loop therefore
    # has deterministic, simple streaming behavior with no sampling or workers.
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

    def take_same_file(self, n: int) -> Tensor:
        while self.tokens.numel() < n or (self.tokens.numel() - self.pos) < n:
            self._advance_file()
        chunk = self.tokens[self.pos : self.pos + n]
        self.pos += n
        return chunk


class DistributedTokenLoader:
    # Each call consumes a contiguous chunk from the shared token stream, then slices out
    # one disjoint span per rank. The extra "+1" token lets us build (x, y) by shifting.
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

    def next_local_span(self, local_tokens_plus_one: int, same_file: bool = False) -> Tensor:
        take_fn = self.stream.take_same_file if same_file else self.stream.take
        chunk = take_fn(local_tokens_plus_one * self.world_size)
        start = self.rank * local_tokens_plus_one
        local = chunk[start : start + local_tokens_plus_one].to(dtype=torch.int64)
        return local.to(self.device, non_blocking=True)

# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (self.weight.shape[0],), weight=self.weight.to(dtype=x.dtype), eps=self.eps)


class CastedLinear(nn.Linear):
    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for low-precision compute.
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)

class BitLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.weight_latent = nn.Parameter(torch.empty(out_features, in_features, dtype=LOW_PRECISION_DTYPE))
        nn.init.normal_(self.weight_latent, std=0.02)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: Tensor) -> Tensor:
        scale = self.weight_latent.detach().abs().mean(dim=1, keepdim=True).clamp(min=1e-5)
        weight_ternary = torch.round(self.weight_latent / scale).clamp(-1, 1)
        w = weight_ternary.detach() - self.weight_latent.detach() + self.weight_latent
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, (w * scale).to(dtype=x.dtype), bias)

class LowRankFastAdapter(nn.Module):
    def __init__(self, d_in: int, d_out: int, rank: int, key: str, gate_init: float = 0.05):
        super().__init__()
        self.key = key
        self.rank = rank
        self.u_basis = nn.Parameter(torch.empty(d_out, rank))
        self.v_basis = nn.Parameter(torch.empty(rank, d_in))
        self.gate = nn.Parameter(torch.tensor(gate_init, dtype=torch.float32))
        self.log_lr = nn.Parameter(torch.tensor(-4.0, dtype=torch.float32))
        self.logit_decay = nn.Parameter(torch.tensor(4.0, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.u_basis, mean=0.0, std=0.02)
        nn.init.normal_(self.v_basis, mean=0.0, std=0.02)

    def lr(self) -> Tensor:
        return F.softplus(self.log_lr)

    def decay(self) -> Tensor:
        return torch.sigmoid(self.logit_decay)

    def delta(self, x: Tensor, fast_state: dict[str, Tensor] | None) -> Tensor:
        dummy = (0.0 * self.log_lr + 0.0 * self.logit_decay + 0.0 * self.gate + 0.0 * self.u_basis.sum() + 0.0 * self.v_basis.sum()).to(dtype=x.dtype)
        if fast_state is None or self.key not in fast_state:
            return torch.zeros((*x.shape[:-1], self.u_basis.shape[0]), device=x.device, dtype=x.dtype) + dummy
        b = fast_state[self.key].to(dtype=x.dtype)
        h = F.linear(x, self.v_basis.to(dtype=x.dtype))
        h = F.linear(h, b)
        h = F.linear(h, self.u_basis.to(dtype=x.dtype))
        return self.gate.to(dtype=x.dtype) * h + dummy

def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    # Keep small/control parameters in fp32 even when the model body runs in low precision.
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    # Caches cos/sin tables per sequence length on the current device.
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        use_fast_adapter: bool = False,
        fast_rank: int = 16,
        fast_key: str = "",
        fast_gate_init: float = 0.05,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = BitLinear(dim, dim, bias=False)
        self.c_k = BitLinear(dim, kv_dim, bias=False)
        self.c_v = BitLinear(dim, kv_dim, bias=False)
        self.proj = BitLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.v_fast = (
            LowRankFastAdapter(dim, kv_dim, fast_rank, fast_key, gate_init=fast_gate_init)
            if use_fast_adapter else None
        )
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor, fast_state: dict[str, Tensor] | None = None) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v_base = self.c_v(x)
        if self.v_fast is not None:
            v_base = v_base + self.v_fast.delta(x, fast_state)
        v = v_base.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    # relu^2 MLP from the original modded-nanogpt setup
    def __init__(
        self,
        dim: int,
        mlp_mult: int,
        use_fast_adapter: bool = False,
        fast_rank: int = 16,
        fast_key: str = "",
        fast_gate_init: float = 0.05,
    ):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = BitLinear(dim, hidden, bias=False)
        self.proj = BitLinear(hidden, dim, bias=False)
        self.proj._zero_init = True
        self.fc_fast = (
            LowRankFastAdapter(dim, hidden, fast_rank, fast_key, gate_init=fast_gate_init)
            if use_fast_adapter else None
        )

    def forward(self, x: Tensor, fast_state: dict[str, Tensor] | None = None) -> Tensor:
        h = self.fc(x)
        if self.fc_fast is not None:
            h = h + self.fc_fast.delta(x, fast_state)
        x = torch.relu(h)
        return self.proj(x.square())

class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        attn: CausalSelfAttention | None = None,
        mlp: MLP | None = None,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.mlp_norm = RMSNorm(dim)
        self.attn = attn if attn is not None else CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = mlp if mlp is not None else MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor, fast_state: dict[str, Tensor] | None = None) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x), fast_state=fast_state)
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x), fast_state=fast_state)
        return x

class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        use_factor_embed: bool,
        embed_dim: int,
        num_unique_attn: int,
        num_unique_mlp: int,
        use_fast_adapters: bool,
        fast_rank: int,
        fast_gate_init: float,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        
        self.use_factor_embed = use_factor_embed
        self.embed_dim = embed_dim
        self.tok_emb = nn.Embedding(vocab_size, embed_dim if use_factor_embed else model_dim)
        self.emb_proj = CastedLinear(embed_dim, model_dim, bias=False) if use_factor_embed else None

        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))

        self.attn_bank = nn.ModuleList([
            CausalSelfAttention(
                model_dim,
                num_heads,
                num_kv_heads,
                rope_base,
                qk_gain_init,
                use_fast_adapter=use_fast_adapters,
                fast_rank=fast_rank,
                fast_key=f"attn_v_{i}",
                fast_gate_init=fast_gate_init,
            )
            for i in range(num_unique_attn)
        ])
        self.mlp_bank = nn.ModuleList([
            MLP(
                model_dim,
                mlp_mult,
                use_fast_adapter=use_fast_adapters,
                fast_rank=fast_rank,
                fast_key=f"mlp_fc_{i}",
                fast_gate_init=fast_gate_init,
            )
            for i in range(num_unique_mlp)
        ])

        self.blocks = nn.ModuleList([
            Block(
                model_dim,
                num_heads,
                num_kv_heads,
                mlp_mult,
                rope_base,
                qk_gain_init,
                attn=self.attn_bank[i % num_unique_attn],
                mlp=self.mlp_bank[i % num_unique_mlp],
            )
            for i in range(num_layers)
        ])
        
        self.final_norm = RMSNorm(model_dim)
        
        self.out_fast = (
            LowRankFastAdapter(model_dim, model_dim, fast_rank, "out", gate_init=fast_gate_init)
            if use_fast_adapters else None
        )

        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
            if getattr(self, "emb_proj", None) is not None:
                nn.init.normal_(self.emb_proj.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if getattr(module, "_zero_init", False):
                if hasattr(module, "weight_latent"):
                    nn.init.zeros_(module.weight_latent)
                elif hasattr(module, "weight"):
                    nn.init.zeros_(module.weight)

    def forward(
        self,
        input_ids: Tensor,
        target_ids: Tensor,
        fast_state: dict[str, Tensor] | None = None,
        loss_mask: Tensor | None = None,
    ) -> Tensor:
        x = self.tok_emb(input_ids)
        if getattr(self, "emb_proj", None) is not None:
            x = self.emb_proj(x)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []

        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0, fast_state=fast_state)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0, fast_state=fast_state)

        x = self.final_norm(x)
        if self.out_fast is not None:
            x = x + self.out_fast.delta(x, fast_state)

        x = x.reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)

        if self.tie_embeddings:
            if getattr(self, "emb_proj", None) is not None:
                z = F.linear(x, self.emb_proj.weight.T.to(dtype=x.dtype))
                logits_proj = F.linear(z, self.tok_emb.weight.to(dtype=x.dtype))
            else:
                logits_proj = F.linear(x, self.tok_emb.weight.to(dtype=x.dtype))
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)

        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        losses = F.cross_entropy(logits.float(), targets, reduction="none")

        if loss_mask is None:
            return losses.mean()

        mask = loss_mask.reshape(-1).to(dtype=losses.dtype)
        denom = mask.sum().clamp_min(1.0)
        return (losses * mask).sum() / denom

def init_fast_state(rank: int, device: torch.device, keys: tuple[str, ...] = FAST_STATE_KEYS) -> dict[str, Tensor]:
    return {k: torch.zeros(rank, rank, device=device, dtype=torch.float32) for k in keys}

def clone_fast_state(state: dict[str, Tensor]) -> dict[str, Tensor]:
    return {k: v.clone() for k, v in state.items()}

def detach_fast_state(state: dict[str, Tensor]) -> dict[str, Tensor]:
    return {k: v.detach() for k, v in state.items()}

def iter_fast_adapters(module: nn.Module) -> list[LowRankFastAdapter]:
    return [m for m in module.modules() if isinstance(m, LowRankFastAdapter)]

def iter_fast_adapter_params(module: nn.Module) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    seen: set[int] = set()
    for adapter in iter_fast_adapters(module):
        for param in adapter.parameters():
            if id(param) in seen:
                continue
            seen.add(id(param))
            params.append(param)
    return params

def mask_grads_to_param_ids(module: nn.Module, allowed_param_ids: set[int]) -> None:
    for param in module.parameters():
        if id(param) not in allowed_param_ids:
            param.grad = None

def adapt_fast_state(
    model: nn.Module,
    fast_state: dict[str, Tensor],
    x_prefix: Tensor,
    y_prefix: Tensor,
    steps: int,
    clip: float,
    create_graph: bool = False,
    detach_result: bool | None = None,
    adapters: tuple[LowRankFastAdapter, ...] | None = None,
) -> dict[str, Tensor]:
    adapters = tuple(iter_fast_adapters(model) if adapters is None else adapters)
    state = {k: v.clone().detach().requires_grad_(True) for k, v in fast_state.items()}
    if detach_result is None:
        detach_result = not create_graph
    active_adapters = tuple(a for a in adapters if a.key in state)
    if not active_adapters or steps <= 0:
        return detach_fast_state(state) if detach_result else state

    for step_idx in range(steps):
        # Recompute the prefix loss from the exact state being adapted.
        # Cached outer losses are often detached from this internal clone,
        # which would make autograd.grad see an unused tensor.
        loss = model(x_prefix, y_prefix, fast_state=state)
        grads = torch.autograd.grad(
            loss,
            [state[a.key] for a in active_adapters],
            create_graph=create_graph,
            allow_unused=False,
        )
        next_state = {}
        for a, g in zip(active_adapters, grads, strict=True):
            g_norm = g.norm()
            if clip > 0:
                g = g * (clip / g_norm.clamp_min(clip))
            next_state[a.key] = a.decay() * state[a.key] - a.lr() * g
        state = next_state

    return detach_fast_state(state) if detach_result else state

# -----------------------------
# TRAINING
# -----------------------------

def main(argv: list[str] | None = None) -> None:
    global LOW_PRECISION_DTYPE, zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    cli_overrides = parse_cli_overrides(sys.argv[1:] if argv is None else argv)
    os.environ.update(cli_overrides)
    args = Hyperparameters()
    if args.model_dim % args.num_heads != 0:
        raise ValueError("MODEL_DIM must be divisible by NUM_HEADS")
    if not 1 <= args.num_unique_attn <= args.num_layers:
        raise ValueError(f"NUM_UNIQUE_ATTN must be in [1, NUM_LAYERS], got {args.num_unique_attn}")
    if not 1 <= args.num_unique_mlp <= args.num_layers:
        raise ValueError(f"NUM_UNIQUE_MLP must be in [1, NUM_LAYERS], got {args.num_unique_mlp}")
    if args.qat_every_steps <= 0:
        raise ValueError(f"QAT_EVERY_STEPS must be positive, got {args.qat_every_steps}")
    if args.train_meta_steps < 0:
        raise ValueError(f"TRAIN_META_STEPS must be non-negative, got {args.train_meta_steps}")
    if args.use_fast_adapters and args.fast_gate_init <= 0.0:
        raise ValueError(f"FAST_GATE_INIT must be positive when fast adapters are enabled, got {args.fast_gate_init}")
    if args.train_meta_min_seqs_per_side <= 0:
        raise ValueError(
            f"TRAIN_META_MIN_SEQS_PER_SIDE must be positive, got {args.train_meta_min_seqs_per_side}"
        )
    if args.train_meta_aux_weight < 0.0:
        raise ValueError(f"TRAIN_META_AUX_WEIGHT must be non-negative, got {args.train_meta_aux_weight}")
    if args.train_meta_loss_a_weight < 0.0 or args.train_meta_loss_b_weight < 0.0:
        raise ValueError(
            "TRAIN_META_LOSS_A_WEIGHT and TRAIN_META_LOSS_B_WEIGHT must be non-negative, "
            f"got {args.train_meta_loss_a_weight} and {args.train_meta_loss_b_weight}"
        )
    if args.train_meta_adapter_only and not args.use_fast_adapters:
        raise ValueError("TRAIN_META_ADAPTER_ONLY requires USE_FAST_ADAPTERS=1")
    if args.train_meta_freeze_backbone and not args.use_fast_adapters:
        raise ValueError("TRAIN_META_FREEZE_BACKBONE requires USE_FAST_ADAPTERS=1")
    if args.init_model_path and not Path(args.init_model_path).exists():
        raise FileNotFoundError(f"INIT_MODEL_PATH not found: {args.init_model_path}")

    # -----------------------------
    # DISTRIBUTED + CUDA SETUP
    # -----------------------------

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1 and "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    if args.train_batch_tokens <= 0 or args.train_seq_len <= 0:
        raise ValueError("TRAIN_BATCH_TOKENS and TRAIN_SEQ_LEN must be positive")
    if args.train_batch_tokens % (8 * args.train_seq_len) != 0:
        raise ValueError(
            f"TRAIN_BATCH_TOKENS={args.train_batch_tokens} must be divisible by 8 * TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    local_train_tokens = args.train_batch_tokens // (world_size * grad_accum_steps)
    meta_pair_tokens = max(local_train_tokens // 2, args.train_meta_min_seqs_per_side * args.train_seq_len)
    meta_local_tokens = 2 * meta_pair_tokens
    if args.use_fast_adapters and args.train_meta_every > 0 and local_train_tokens % args.train_seq_len != 0:
        raise ValueError(
            "Fast-adapter meta-training requires each local microbatch to contain an integral number of sequences; "
            f"got local_tokens={local_train_tokens} and TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    LOW_PRECISION_DTYPE = resolve_low_precision_dtype(device)
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    # Fast math knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    use_cudnn_sdp = False
    use_flash_sdp = torch.cuda.get_device_capability(device)[0] >= 8
    use_mem_efficient_sdp = True
    use_math_sdp = True
    enable_cudnn_sdp(use_cudnn_sdp)
    enable_flash_sdp(use_flash_sdp)
    enable_mem_efficient_sdp(use_mem_efficient_sdp)
    enable_math_sdp(use_math_sdp)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(f"low_precision_dtype:{str(LOW_PRECISION_DTYPE).removeprefix('torch.')}")
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    # -----------------------------
    # TOKENIZER + VALIDATION METRIC SETUP
    # -----------------------------

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    # -----------------------------
    # MODEL + OPTIMIZER SETUP
    # -----------------------------

    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        use_factor_embed=args.use_factor_embed,
        embed_dim=args.embed_dim,
        num_unique_attn=args.num_unique_attn,
        num_unique_mlp=args.num_unique_mlp,
        use_fast_adapters=args.use_fast_adapters,
        fast_rank=args.fast_rank,
        fast_gate_init=args.fast_gate_init,
    ).to(device=device, dtype=LOW_PRECISION_DTYPE)
    if args.init_model_path:
        loaded_obj = torch.load(args.init_model_path, map_location="cpu")
        if isinstance(loaded_obj, dict) and "state_dict" in loaded_obj and isinstance(loaded_obj["state_dict"], dict):
            init_state = loaded_obj["state_dict"]
        else:
            init_state = loaded_obj
        load_result = base_model.load_state_dict(init_state, strict=args.init_model_strict)
        if not args.init_model_strict:
            missing = ", ".join(load_result.missing_keys[:8]) if load_result.missing_keys else ""
            unexpected = ", ".join(load_result.unexpected_keys[:8]) if load_result.unexpected_keys else ""
            log0(
                f"init_model_path:{args.init_model_path} init_model_strict:{args.init_model_strict} "
                f"missing_keys:{len(load_result.missing_keys)}"
                + (f" sample_missing:{missing}" if missing else "")
                + f" unexpected_keys:{len(load_result.unexpected_keys)}"
                + (f" sample_unexpected:{unexpected}" if unexpected else "")
            )
        else:
            log0(f"init_model_path:{args.init_model_path} init_model_strict:{args.init_model_strict}")
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    fast_state_keys = build_fast_state_keys(args.num_unique_attn, args.num_unique_mlp) if args.use_fast_adapters else ()
    fast_adapters = tuple(iter_fast_adapters(base_model)) if args.use_fast_adapters else ()
    fast_adapter_params = tuple(iter_fast_adapter_params(base_model)) if args.use_fast_adapters else ()
    fast_adapter_param_ids = {id(p) for p in fast_adapter_params}
    zero_fast_state_template = (
        init_fast_state(args.fast_rank, device, fast_state_keys)
        if fast_adapters
        else None
    )

    if args.use_compile:
        compiled_model = torch.compile(
            base_model,
            dynamic=False,
            fullgraph=args.compile_fullgraph,
        )
    else:
        compiled_model = base_model

    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    # Optimizer split:
    # - token embedding (Adam) uses EMBED_LR
    # - untied lm_head (Adam) uses HEAD_LR
    # - matrix params in transformer blocks use MATRIX_LR via Muon
    # - vectors/scalars use SCALAR_LR via Adam
    seen = set()
    matrix_params = []
    scalar_params = []

    special = {id(base_model.tok_emb.weight)}
    if getattr(base_model, "emb_proj", None) is not None:
        special.add(id(base_model.emb_proj.weight))
    if base_model.lm_head is not None:
        special.add(id(base_model.lm_head.weight))

    for name, p in base_model.named_parameters():
        if id(p) in seen or id(p) in special:
            continue
        seen.add(id(p))
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS):
            matrix_params.append(p)
        else:
            scalar_params.append(p)

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    tok_group_params = [base_model.tok_emb.weight]
    if getattr(base_model, "emb_proj", None) is not None:
        tok_group_params.append(base_model.emb_proj.weight)

    optimizer_tok = torch.optim.Adam(
        [{"params": tok_group_params, "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0(
        f"sdp_backends:cudnn={use_cudnn_sdp} flash={use_flash_sdp} "
        f"mem_efficient={use_mem_efficient_sdp} math={use_math_sdp}"
    )
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    if args.use_fast_adapters:
        log0(
            f"fast_gate_init:{args.fast_gate_init:.4f} "
            f"meta_pair_tokens:{meta_pair_tokens} meta_local_tokens:{meta_local_tokens} "
            f"train_meta_min_seqs_per_side:{args.train_meta_min_seqs_per_side} "
            f"train_meta_same_shard:{args.train_meta_same_shard}"
        )
        if args.train_meta_every > 0 and args.train_meta_steps > 0:
            log0(
                f"train_meta_aux_weight:{args.train_meta_aux_weight:.4f} "
                f"train_meta_loss_a_weight:{args.train_meta_loss_a_weight:.4f} "
                f"train_meta_loss_b_weight:{args.train_meta_loss_b_weight:.4f} "
                f"train_meta_adapter_only:{args.train_meta_adapter_only} "
                f"train_meta_freeze_backbone:{args.train_meta_freeze_backbone}"
            )
    log0(f"seed:{args.seed}")

    # -----------------------------
    # DATA LOADER & MODEL WARMUP
    # -----------------------------

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    # Warmup primes the compiled forward/backward/optimizer paths, then we restore the
    # initial weights/optimizer state so measured training starts from the true init.
    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=LOW_PRECISION_DTYPE, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # -----------------------------
    # MAIN TRAINING LOOP
    # -----------------------------

    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (
            args.val_loss_every > 0
            and (args.validate_at_step_zero or step > 0)
            and step % args.val_loss_every == 0
        )
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
                fast_state_keys=fast_state_keys,
                adapters=fast_adapters,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        progress = step / max(args.iterations, 1)
        meta_phase_active = (
            args.use_fast_adapters
            and args.train_meta_start_frac <= progress < args.train_meta_end_frac
        )
        meta_active = (
            meta_phase_active
            and args.train_meta_every > 0
            and args.train_meta_steps > 0
            and (step % args.train_meta_every == 0)
        )
        freeze_backbone_active = args.train_meta_freeze_backbone and meta_phase_active and bool(fast_adapter_param_ids)

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        qat_active = args.use_projection_qat and (elapsed_ms >= args.qat_start_ms)
        scale = lr_mul(step, elapsed_ms)
        if qat_active:
            scale *= args.qat_lr_scale
            
        if meta_active:
            zero_grad_all()
            train_loss = torch.zeros((), device=device)
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1

                x_main, y_main = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)

                pair_tokens = meta_pair_tokens
                local = train_loader.next_local_span(meta_local_tokens + 1, same_file=args.train_meta_same_shard)
                x_a = local[:pair_tokens].reshape(-1, args.train_seq_len)
                y_a = local[1 : pair_tokens + 1].reshape(-1, args.train_seq_len)
                x_b = local[pair_tokens : 2 * pair_tokens].reshape(-1, args.train_seq_len)
                y_b = local[pair_tokens + 1 : 2 * pair_tokens + 1].reshape(-1, args.train_seq_len)

                zero_state = clone_fast_state(zero_fast_state_template) if zero_fast_state_template is not None else {}
                with torch.autocast(device_type="cuda", dtype=LOW_PRECISION_DTYPE, enabled=True):
                    loss_main = model(x_main, y_main)
                    loss_a = model(x_a, y_a, fast_state=zero_state)
                    fast_state_1 = adapt_fast_state(
                        model,
                        zero_state,
                        x_a,
                        y_a,
                        steps=args.train_meta_steps,
                        clip=args.fast_grad_clip,
                        create_graph=not args.train_meta_first_order,
                        detach_result=False,
                        adapters=fast_adapters,
                    )
                    loss_b = model(x_b, y_b, fast_state=fast_state_1)
                    meta_loss = args.train_meta_loss_a_weight * loss_a + args.train_meta_loss_b_weight * loss_b
                    loss = loss_main + args.train_meta_aux_weight * meta_loss

                train_loss += loss.detach()
                (loss_main * grad_scale).backward()
                if args.train_meta_aux_weight > 0.0:
                    if args.train_meta_adapter_only and fast_adapter_params:
                        meta_grads = torch.autograd.grad(
                            args.train_meta_aux_weight * meta_loss * grad_scale,
                            fast_adapter_params,
                            allow_unused=True,
                        )
                        for param, grad in zip(fast_adapter_params, meta_grads, strict=True):
                            if grad is None:
                                continue
                            grad = grad.detach()
                            if param.grad is None:
                                param.grad = grad
                            else:
                                param.grad.add_(grad)
                    else:
                        (args.train_meta_aux_weight * meta_loss * grad_scale).backward()
                if freeze_backbone_active:
                    mask_grads_to_param_ids(base_model, fast_adapter_param_ids)

            train_loss /= grad_accum_steps
        else:
            zero_grad_all()
            train_loss = torch.zeros((), device=device)
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=LOW_PRECISION_DTYPE, enabled=True):
                    loss = model(x, y)
                train_loss += loss.detach()
                (loss * grad_scale).backward()
                if freeze_backbone_active:
                    mask_grads_to_param_ids(base_model, fast_adapter_param_ids)
            train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()
        
        if qat_active and step % args.qat_every_steps == 0:
            project_model_to_export_grid(base_model)

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        log_all_steps = args.iterations <= args.train_log_every
        should_log_train = (
            args.train_log_every > 0
            and (
                log_all_steps
                or step <= 10
                or step % args.train_log_every == 0
                or step == args.iterations
                or stop_after_step is not None
            )
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"meta_active:{meta_active} meta_phase_active:{meta_phase_active} "
                f"freeze_backbone_active:{freeze_backbone_active} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )

        # Needed to sync whether we've reached the wallclock cap.
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    # -----------------------------
    # SERIALIZATION + ROUNDTRIP VALIDATION
    # -----------------------------
    # Save the raw state (useful for debugging/loading in PyTorch directly), then always produce
    # the compressed int8+zlib artifact and validate the round-tripped weights.

    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    quant_raw = pack_pgq2(quant_obj)
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_raw_bytes = len(quant_raw)
    if master_process:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int8.ptz")
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"Serialized model int8+zlib: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = unpack_pgq2(zlib.decompress(quant_blob_disk))
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args,
        model,
        rank,
        world_size,
        device,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        fast_state_keys=fast_state_keys,
        adapters=fast_adapters,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main(sys.argv[1:])
