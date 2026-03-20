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

## Phase 9: Meta Investigation and Reproduction

Meta was revisited because two early hypotheses looked plausible:

1. the fast adapters were dead at initialization due to `gate=0`
2. the support/query split was too small to define a useful adaptation task

### Fixes added to the script

- `FAST_GATE_INIT` was introduced so fast adapters no longer start fully closed
- `TRAIN_META_MIN_SEQS_PER_SIDE` was introduced so support/query spans could be larger than a single `64`-token sequence
- later, `TRAIN_META_MODE=replace|aux` was added so the legacy meta objective could be reproduced explicitly

### What did not work

- eval-time TTT still showed no measurable benefit in direct A/B tests
- stage-2 adapter-only meta on top of a pretrained backbone also showed no gain
- auxiliary meta loss and late-phase backbone freezing reduced collapse, but they did not beat the legacy from-scratch meta path

### What did work

The legacy replace-style meta objective turned out to be real and reproducible.

Configuration:

- `USE_FAST_ADAPTERS=1`
- `FAST_RANK=2`
- `FAST_GRAD_CLIP=0.1`
- `FAST_GATE_INIT=0.05`
- `TRAIN_META_MODE=replace`
- `TRAIN_META_EVERY=4`
- `TRAIN_META_START_FRAC=0.5`
- `TRAIN_META_END_FRAC=1.0`
- `TRAIN_META_STEPS=1`
- `TRAIN_META_FIRST_ORDER=1`
- `TRAIN_META_MIN_SEQS_PER_SIDE=8`
- `TRAIN_META_LOSS_A_WEIGHT=0.2`
- `TRAIN_META_LOSS_B_WEIGHT=0.8`
- `TRAIN_META_ADAPTER_ONLY=0`
- `TRAIN_META_FREEZE_BACKBONE=0`
- `TRAIN_META_SAME_SHARD=0`
- `EVAL_TTT_STEPS=0`

### `1000`-step meta-replace results

| Seed | `val_bpb` | Final int8 `val_bpb` |
|---:|---:|---:|
| `1337` | `2.8012` | `2.8020` |
| `2024` | `2.8217` | `2.8229` |
| `3407` | `2.7627` | `2.7633` |

Mean `val_bpb`: `2.7952`

Conclusion:

- Meta remains non-useful as test-time adaptation.
- But the legacy replace-style meta objective improves training enough to beat the no-meta backbone.
- The gain appears to come from the training trajectory, not from runtime TTT.

### Completed schedule sweep on `SEED=1337`

These runs held the rest of the meta-replace recipe fixed and only changed the schedule or local meta context:

| Variant | `val_bpb` | Notes |
|---|---:|---|
| `START=0.5, EVERY=4, CTX=8` | `2.8012` | original replace baseline |
| `START=0.4, EVERY=4, CTX=8` | `2.8430` | clearly worse |
| `START=0.6, EVERY=4, CTX=8` | `2.8095` | slightly worse |
| `START=0.5, EVERY=2, CTX=8` | `2.7903` | better than baseline |
| `START=0.5, EVERY=8, CTX=8` | `2.7904` | effectively tied with `EVERY=2` |
| `START=0.5, EVERY=4, CTX=12` | `2.7763` | best fixed-seed schedule result |

Takeaways:

- `TRAIN_META_START_FRAC=0.5` remains the best start point.
- `TRAIN_META_EVERY=2` and `8` both improved over `4` when `CTX=8`.
- Increasing the local meta context from `8` to `12` sequences per side gave the strongest single fixed-seed gain.

### Combined follow-up tests

The schedule gains did not combine cleanly:

| Variant | `val_bpb` |
|---|---:|
| `CTX=12, EVERY=4` | `2.7763` |
| `CTX=12, EVERY=2` | `2.7831` |
| `CTX=12, EVERY=8` | `2.8189` |

This indicates a real interaction:

- larger meta context helps
- more frequent meta helps at `CTX=8`
- but combining larger context with `EVERY=2` does not beat `CTX=12, EVERY=4`
- `EVERY=8` becomes too sparse once the local meta context is larger

### `100`-step coarse screen for fast rank and loss weights

Using `CTX=12, EVERY=4` as the short-run baseline:

| Variant | `val_bpb` | Outcome |
|---|---:|---|
| `FAST_RANK=2`, loss `0.2 / 0.8` | `3.2697` | baseline |
| `FAST_RANK=1`, loss `0.2 / 0.8` | `3.6178` | unstable and much worse |
| `FAST_RANK=4`, loss `0.2 / 0.8` | `3.5189` | unstable and much worse |
| `FAST_RANK=2`, loss `0.1 / 0.9` | `3.2957` | worse |
| `FAST_RANK=2`, loss `0.0 / 1.0` | `3.2794` | slightly worse |

Short-run verdict:

- `FAST_RANK=2` remains the only sensible setting from these tests
- `TRAIN_META_LOSS_A_WEIGHT=0.2` and `TRAIN_META_LOSS_B_WEIGHT=0.8` remain the best loss split tested
- none of these short-run screens justified promotion to the long-horizon main line

### `CTX=12` three-seed comparison

The `CTX=12, EVERY=4` line was checked across the same three seeds as the original `CTX=8, EVERY=4` main line:

| Seed | `CTX=8, EVERY=4` | `CTX=12, EVERY=4` |
|---:|---:|---:|
| `1337` | `2.8012` | `2.7763` |
| `2024` | `2.8217` | `2.8337` |
| `3407` | `2.7627` | `2.7678` |

Means:

- `CTX=8, EVERY=4`: `2.7952`
- `CTX=12, EVERY=4`: `2.7926`

Interpretation:

- `CTX=12` has a slightly better three-seed mean
- but it only wins on `1/3` seeds
- it loses on the strongest single seed (`3407`)
- and it also loses on `2024`

Because the mean advantage is tiny and the best single run remains `CTX=8`, `CTX=12` was **not** promoted to the main line.

## Current Winning Configuration

This remains the current primary Moonshot configuration from the experiments so far:

```bash
RUN_ID=moonshot_2x1_fe96_meta_replace_ctx8_1000 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
SEED=3407 \
VOCAB_SIZE=1024 \
NUM_LAYERS=2 \
NUM_UNIQUE_ATTN=2 \
NUM_UNIQUE_MLP=1 \
MODEL_DIM=128 \
NUM_HEADS=4 \
NUM_KV_HEADS=2 \
MLP_MULT=2 \
USE_FAST_ADAPTERS=1 \
FAST_RANK=2 \
FAST_GRAD_CLIP=0.1 \
FAST_GATE_INIT=0.05 \
TRAIN_META_MODE=replace \
TRAIN_META_EVERY=4 \
TRAIN_META_START_FRAC=0.5 \
TRAIN_META_END_FRAC=1.0 \
TRAIN_META_STEPS=1 \
TRAIN_META_FIRST_ORDER=1 \
TRAIN_META_MIN_SEQS_PER_SIDE=8 \
TRAIN_META_LOSS_A_WEIGHT=0.2 \
TRAIN_META_LOSS_B_WEIGHT=0.8 \
TRAIN_META_ADAPTER_ONLY=0 \
TRAIN_META_FREEZE_BACKBONE=0 \
TRAIN_META_SAME_SHARD=0 \
TRAIN_SEQ_LEN=64 \
TRAIN_BATCH_TOKENS=1024 \
ITERATIONS=1000 \
WARMUP_STEPS=5 \
WARMDOWN_ITERS=0 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=10 \
MAX_WALLCLOCK_SECONDS=0 \
VAL_TOKENS_LIMIT=2048 \
EVAL_ADAPT=256 \
EVAL_SCORE=64 \
EVAL_TTT_STEPS=0 \
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

- `SEED=3407`
- `ITERATIONS=1000`
- `val_loss=5.1003`
- `val_bpb=2.7627`
- final int8 roundtrip `val_bpb=2.7633`

### Multi-seed summary for the winning config at `1000` steps

| Seed | `val_bpb` | Final int8 `val_bpb` |
|---:|---:|---:|
| `1337` | `2.8012` | `2.8020` |
| `2024` | `2.8217` | `2.8229` |
| `3407` | `2.7627` | `2.7633` |

Mean `val_bpb`: `2.7952`

Why this remains the main line:

- it still owns the best single observed run
- the `CTX=12` follow-up did not beat it on `SEED=3407`
- the `CTX=12` mean advantage was too small and too seed-dependent to justify switching

## Main Conclusions

1. The winning direction is the Moonshot backbone, not the baseline script.
2. The strongest gains came from:
   - partial sharing
   - factorized embeddings
   - retuned optimizer settings
3. The most reliable sharing pattern so far is:
   - unique attention
   - shared MLP
4. Eval-time TTT has still not shown measurable value in the tested regime.
5. The tuned Moonshot quantizes much better than the baseline:
   - baseline final int8 metric degrades noticeably
   - Moonshot final int8 metric is almost unchanged
6. The current best result comes from replace-style meta as a training objective:
   - no-meta `1000` steps, `SEED=1337`: `2.8297`
   - meta-replace `1000` steps, `SEED=1337`: `2.8012`
   - meta-replace `1000` steps, `SEED=3407`: `2.7627`
7. The meta gain does not appear to come from runtime adaptation:
   - stage-2 frozen-backbone meta did not improve a pretrained backbone
   - `EVAL_TTT_STEPS` did not improve validation
   - the benefit comes from how the replace-style meta objective shapes full training

## New Moonshot Meta Path

A new Moonshot-only meta-training path has been implemented in `train_gpt_moonshot.py` to better match the original meta-learning goal.

What it does:

- keeps the existing Moonshot fast-adapter / fast-state mechanism
- replaces flat support/query token pairs with BOS-delimited document episodes when enabled
- adapts on a support prefix from a single document
- scores the outer loss only on a held-out continuation from that same document

New controls:

- `TRAIN_META_DOCUMENT_EPISODES=1`
- `TRAIN_META_BATCH_DOCS`
- `TRAIN_META_ADAPT_TOKENS`
- `TRAIN_META_QUERY_TOKENS`
- `TRAIN_META_DOC_BOS_ID`

Interpretation:

- this is much closer to the intended meta-learning setup than the original flat-stream replace objective
- it is still a Moonshot training-path change, not yet a validated new winner
- no leaderboard-quality benchmark has been recorded for this mode yet in the experiment log

Status:

- implemented locally in `train_gpt_moonshot.py`
- compiles successfully
- still needs real GPU experiments before promotion into the main configuration

## Recommended Next Step

After completing the schedule sweep and the `CTX=12` follow-up, the next meaningful experiments are:

1. More seed checks on the `CTX=8, EVERY=4` main line if the goal is leaderboard robustness.
2. A longer horizon on the best single-seed recipe (`SEED=3407`, `CTX=8`, `EVERY=4`) if wallclock budget allows.
3. No further short-run tuning on `FAST_RANK` or the `loss_a/loss_b` split unless a new structural idea changes the meta objective itself.
