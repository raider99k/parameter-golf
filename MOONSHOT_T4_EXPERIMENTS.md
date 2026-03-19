# Moonshot Colab T4 Experiment Summary

This file summarizes the Colab T4 experiments run against `train_gpt.py` and `train_gpt_moonshot.py`, the conclusions from each phase, and the current winning Moonshot configuration.

All results below come from the Colab commands and logs captured in the working chat thread. They are not reconstructed from local log files.

## Environment

- Hardware: Colab T4
- Dataset: `./data/datasets/fineweb10B_sp1024`
- Tokenizer: `./data/tokenizers/fineweb_1024_bpe.model`
- Vocab size: `1024`
- Common short-run settings unless noted:
  - `NUM_LAYERS=2`
  - `MODEL_DIM=128`
  - `NUM_HEADS=4`
  - `NUM_KV_HEADS=2`
  - `MLP_MULT=2`
  - `TRAIN_SEQ_LEN=64`
  - `TRAIN_BATCH_TOKENS=1024`

## Important Evaluation Notes

- `train_gpt.py` does not use `EVAL_WINDOW`, `EVAL_ADAPT`, or `EVAL_SCORE`. It uses `EVAL_CTX_LEN` and `EVAL_PRED_LEN` instead.
- `train_gpt_moonshot.py` parses `EVAL_WINDOW`, but the current implementation does not use it in validation.
- Moonshot eval-time fast adaptation only runs when both of these are true:
  - `USE_FAST_ADAPTERS=1`
  - `EVAL_TTT_STEPS > 0`
- Early comparisons were directionally useful but not perfectly matched until baseline eval was switched to:
  - `EVAL_CTX_LEN=128`
  - `EVAL_PRED_LEN=64`

## Phase 1: Initial Smoke Runs

### Initial comparison

- Baseline smoke:
  - `train_gpt.py`
  - `TRAIN_SEQ_LEN=128`
  - `val_bpb=14.3133`
- Moonshot smoke:
  - `train_gpt_moonshot.py`
  - `MODEL_DIM=96`
  - `TRAIN_SEQ_LEN=64`
  - adapters and meta enabled
  - `val_bpb=14.8933`

Conclusion:

- Baseline looked better.
- The comparison was not clean because model width, sequence length, and eval path differed.

### Aligned 20-step smokes

After matching `MODEL_DIM=128`, `TRAIN_SEQ_LEN=64`, `WARMUP_STEPS=5`, and related smoke settings:

- Baseline: `val_bpb=14.5551`
- Moonshot, adapters + meta, `NUM_UNIQUE_ATTN=2`, `NUM_UNIQUE_MLP=2`: `val_bpb=16.4039`
- Moonshot, adapters, no meta, `NUM_UNIQUE_ATTN=2`, `NUM_UNIQUE_MLP=2`: `val_bpb=16.3183`

Conclusion:

- That Moonshot setup was clearly worse.
- At this point it became clear that `NUM_UNIQUE_ATTN=2` and `NUM_UNIQUE_MLP=2` with `NUM_LAYERS=2` meant there was no sharing benefit at all.

## Phase 2: Shared Backbone vs Adapters / Meta

Baseline eval was then aligned more closely with:

- `EVAL_CTX_LEN=128`
- `EVAL_PRED_LEN=64`

Aligned baseline smoke:

- Baseline: `val_bpb=14.5695`

Then Moonshot was tested with full sharing:

- Moonshot, `NUM_UNIQUE_ATTN=1`, `NUM_UNIQUE_MLP=1`, adapters + meta:
  - `val_bpb=16.1917`
- Moonshot, `NUM_UNIQUE_ATTN=1`, `NUM_UNIQUE_MLP=1`, adapters only:
  - `val_bpb=15.9645`
- Moonshot, `NUM_UNIQUE_ATTN=1`, `NUM_UNIQUE_MLP=1`, no adapters, no meta:
  - `val_bpb=13.9197`

Then `EVAL_TTT_STEPS=1` was enabled:

- adapters + meta: still `val_bpb=16.1917`
- adapters only: still `val_bpb=15.9645`

Conclusion:

- The shared Moonshot backbone was promising.
- Fast adapters were harmful.
- Meta training did not rescue the adapter path.
- The real gain came from sharing, not from online adaptation.

## Phase 3: 100-Step Runs and Factorized Embeddings

### Baseline at 100 steps

- Baseline `train_gpt.py`: `val_bpb=6.4901`

### First factorized-embedding attempt

Moonshot was then tried with factorized embeddings:

- `USE_FACTOR_EMBED=1`
- `EMBED_DIM=64`

Results:

- adapters + meta: diverged to `NaN` around step 41
- no adapters, no meta: also diverged to `NaN` around step 41

Conclusion:

- The failure was not specific to meta.
- The factorized-embedding path was unstable under the original optimizer settings.

### Stabilized factorized-embedding setup

The following changes stabilized the run:

- `USE_FACTOR_EMBED=1`
- `EMBED_DIM=96`
- `TIED_EMBED_LR=0.001`
- `MATRIX_LR=0.03`
- `GRAD_CLIP_NORM=1.0`

100-step results:

- Moonshot, no adapters, no meta:
  - `val_bpb=3.4321`
- Moonshot, adapters + meta:
  - `val_bpb=3.6074`

Conclusion:

- Factorized embeddings were a large win once retuned.
- Meta remained slower and worse.

## Phase 4: 500-Step Architecture Sweep

Baseline at 500 steps:

- Baseline:
  - `val_bpb=5.3891`
  - final int8 roundtrip `val_bpb=5.5635`

Tuned Moonshot variants at 500 steps:

| Variant | Params | `val_bpb` | Final int8 `val_bpb` | Train time |
|---|---:|---:|---:|---:|
| `1x1 fe96` | `227,076` | `2.9900` | `2.9907` | `58.1s` |
| `2x2 fe96` | `341,768` | `2.9987` | `2.9996` | `60.3s` |
| `1x2 fe96` | `292,612` | `2.9697` | `2.9707` | `57.6s` |
| `2x1 fe96` | `276,232` | `2.9608` | `2.9614` | `60.7s` |
| `1x1 fe96` sparse meta | `228,749` | `3.0779` | `3.0803` | `83.0s` |

Naming convention:

- `2x1` means:
  - `NUM_UNIQUE_ATTN=2`
  - `NUM_UNIQUE_MLP=1`
- `1x2` means:
  - `NUM_UNIQUE_ATTN=1`
  - `NUM_UNIQUE_MLP=2`

Conclusion:

- Best quality at 500 steps came from `2x1 fe96`.
- `1x2 fe96` was very close on one seed, but meta remained worse than no-meta.
- All tuned Moonshot variants were dramatically better than the baseline.

## Phase 5: Seed Stability Sweep

The main question became whether `2x1 fe96` was a real winner or a seed artifact.

### `2x1 fe96`

- `SEED=1337`: `2.9608`
- `SEED=2024`: `3.0778`
- `SEED=3407`: `2.9493`

Mean `val_bpb`: `2.9960`

### `1x2 fe96`

- `SEED=1337`: `2.9697`
- `SEED=2024`: `3.0322`
- `SEED=3407`: `3.2956`

Mean `val_bpb`: `3.0992`

Conclusion:

- `2x1 fe96` is better on average and more stable.
- `1x2 fe96` has larger seed sensitivity.
- `2x1 fe96` became the main line.

## Phase 6: Embedding-Dimension Sweep on `2x1`

All runs below used the tuned optimizer and `SEED=1337`.

| `EMBED_DIM` | `val_bpb` |
|---:|---:|
| `80` | `3.0954` |
| `96` | `2.9608` |
| `128` | `3.0090` |

Conclusion:

- `EMBED_DIM=96` is the best of the tested values.

## Phase 7: Optimizer Sweep on `2x1 fe96`

Reference configuration:

- `TIED_EMBED_LR=0.001`
- `MATRIX_LR=0.03`
- `SCALAR_LR=0.04`
- `GRAD_CLIP_NORM=1.0`

Sweep results:

| Change | `val_bpb` | Conclusion |
|---|---:|---|
| baseline tuned setting | `2.9608` | best |
| `TIED_EMBED_LR=0.0005` | `3.1298` | worse |
| `TIED_EMBED_LR=0.002` | `3.9124` | much worse / unstable |
| `MATRIX_LR=0.025` | `2.9608` | no measurable gain |
| `MATRIX_LR=0.035` | `2.9608` | no measurable gain |
| `SCALAR_LR=0.02` | `2.9931` | slightly worse |
| `SCALAR_LR=0.03` | `3.0717` | worse |

Conclusion:

- The original tuned optimizer remained the best tested setting.

## Phase 8: 1000-Step Scale Check

The best no-meta Moonshot configuration was then extended from `500` to `1000` steps.

Configuration:

- `NUM_UNIQUE_ATTN=2`
- `NUM_UNIQUE_MLP=1`
- `USE_FAST_ADAPTERS=0`
- `USE_FACTOR_EMBED=1`
- `EMBED_DIM=96`
- `TIED_EMBED_LR=0.001`
- `MATRIX_LR=0.03`
- `SCALAR_LR=0.04`
- `GRAD_CLIP_NORM=1.0`
- `SEED=1337`

Result at `1000` steps:

- `val_loss=5.2241`
- `val_bpb=2.8297`
- final int8 roundtrip `val_bpb=2.8290`

Comparison against the same configuration at `500` steps:

| Iterations | `val_bpb` | Final int8 `val_bpb` |
|---:|---:|---:|
| `500` | `2.9608` | `2.9614` |
| `1000` | `2.8297` | `2.8290` |

Conclusion:

- The winning Moonshot backbone continues to improve with longer training.
- There was no late instability in this `1000`-step run.
- The int8 roundtrip remained effectively lossless.

## Current Winning Configuration

This is the current winning Moonshot configuration from the experiments so far:

```bash
RUN_ID=moonshot_2x1_fe96_best \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
SEED=1337 \
VOCAB_SIZE=1024 \
NUM_LAYERS=2 \
NUM_UNIQUE_ATTN=2 \
NUM_UNIQUE_MLP=1 \
MODEL_DIM=128 \
NUM_HEADS=4 \
NUM_KV_HEADS=2 \
MLP_MULT=2 \
USE_FAST_ADAPTERS=0 \
TRAIN_META_EVERY=0 \
TRAIN_SEQ_LEN=64 \
TRAIN_BATCH_TOKENS=1024 \
ITERATIONS=1000 \
WARMUP_STEPS=5 \
WARMDOWN_ITERS=0 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=20 \
MAX_WALLCLOCK_SECONDS=0 \
VAL_TOKENS_LIMIT=2048 \
EVAL_ADAPT=128 \
EVAL_SCORE=64 \
USE_FACTOR_EMBED=1 \
EMBED_DIM=96 \
TIED_EMBED_LR=0.001 \
MATRIX_LR=0.03 \
SCALAR_LR=0.04 \
GRAD_CLIP_NORM=1.0 \
USE_COMPILE=0 \
python train_gpt_moonshot.py
```

### Best observed score with this family

- `SEED=1337`
- `ITERATIONS=1000`
- `val_loss=5.2241`
- `val_bpb=2.8297`
- final int8 roundtrip `val_bpb=2.8290`

### Multi-seed summary for the winning config at `500` steps

| Seed | `val_bpb` | Final int8 `val_bpb` |
|---:|---:|---:|
| `1337` | `2.9608` | `2.9614` |
| `2024` | `3.0778` | `3.0797` |
| `3407` | `2.9493` | `2.9504` |

Mean `val_bpb`: `2.9960`

## Main Conclusions

1. The winning direction is the Moonshot backbone, not the baseline script.
2. The strongest gains came from:
   - partial sharing
   - factorized embeddings
   - retuned optimizer settings
3. The most reliable sharing pattern so far is:
   - unique attention
   - shared MLP
4. Fast adapters and meta training have not helped in the tested regime.
5. The tuned Moonshot quantizes much better than the baseline:
   - baseline final int8 metric degrades noticeably
   - Moonshot final int8 metric is almost unchanged
6. Longer training is still paying off for the no-meta Moonshot backbone:
   - `500` steps: `2.9608`
   - `1000` steps: `2.8297`

## Suggested Next Step

The next meaningful experiment is a matched-horizon baseline comparison, or another `1000`-step Moonshot run on a different seed, to determine how much of the remaining spread is true seed variance versus recoverable with longer training.
