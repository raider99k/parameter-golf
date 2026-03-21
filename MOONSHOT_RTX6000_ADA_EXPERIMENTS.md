# MOONSHOT RTX 6000 Ada Experiments

This document tracks the post-Ampere search phase after moving from the RTX A5000 to the RTX 6000 Ada.

The A5000 work remains documented separately in:

- `MOONSHOT_A5000_EXPERIMENTS.md`

## Starting Point

The carry-over best line from the final A5000 phase is:

- script: `train_gpt_competitive.py`
- `NUM_LAYERS=6`
- `NUM_UNIQUE_ATTN=2`
- `NUM_UNIQUE_MLP=1`
- `MODEL_DIM=384`
- `EMBED_DIM=288`
- `MLP_MULT=3`
- `TRAIN_META_MODE=replace`
- `TRAIN_META_START_FRAC=0.35`
- `TRAIN_META_EVERY=4`
- `USE_FACTOR_EMBED=1`
- `USE_COMPILE=0`

Best observed A5000 result on `SEED=3407`:

- params: `2,197,268`
- float `val_bpb`: `2.3273`
- final int8 roundtrip `val_bpb`: `2.3290`
- total submission size int8+zlib: `2,411,468` bytes

## RTX 6000 Ada Goals

1. verify how much of the improvement comes from the hardware/numeric path
2. continue scaling from the best A5000 backbone
3. avoid mixing hardware effects with too many architecture changes at once
4. keep seed sweeps deferred unless a new line clearly wins

## Immediate Protocol

1. run one carry-over validation of the A5000 best line on RTX 6000 Ada
2. run the next scaling candidates from `train_gpt_competitive.py`
3. only after a clear win, do larger-val confirmation and additional seeds

## First RTX 6000 Ada Wave

All runs below used `SEED=3407`.

### Carry-over baseline and first Ada branches

| Variant | Params | Float `val_bpb` | Final int8 `val_bpb` | Total submission size int8+zlib |
|---|---:|---:|---:|---:|
| carry-over A5000 best: `6L 384d e288`, `MLP_MULT=3` | `2,197,268` | `2.3466` | `2.3475` | `2,410,093` bytes |
| width branch: `6L 448d e320`, `MLP_MULT=2` | `2,503,188` | `2.3567` | `2.3582` | `2,613,368` bytes |
| depth + uniqueness branch: `8L 384d e288`, `4x2`, `MLP_MULT=2` | `3,385,765` | `2.3373` | `2.3364` | `2,759,966` bytes |

### Comparison against the final A5000 best line

Final A5000 best line:

- `6L 384d e288`, `MLP_MULT=3`
- final int8 roundtrip `2.3290`

Observed Ada differences in the first wave:

- the same carry-over line got worse on Ada:
  - `2.3290` on A5000
  - `2.3475` on RTX 6000 Ada
- the first pure width branch also did not win:
  - `6L 448d e320` reached `2.3582`
- the best Ada result from the first wave was the deeper uniqueness-relieved line:
  - `8L 384d e288 4x2` reached `2.3364`

### Interpretation

1. RTX 6000 Ada improved throughput substantially, but did **not** automatically improve the metric on the carry-over recipe.
2. The hardware/numeric path appears to have shifted again:
   - A5000 had already diverged strongly from T4
   - RTX 6000 Ada does not simply continue that trajectory upward
3. The first Ada wave suggests deeper models with more uniqueness may now be relatively more attractive than the pure width-led A5000 winner.
4. Width alone is no longer the obvious next move on Ada:
   - `6L 448d e320` was worse than both the carry-over baseline and the `8L 4x2` branch.
5. The best next search direction on Ada should branch from:
   - `8L 384d e288 4x2`
   - not from `6L 448d e320`

## Notes

- The extreme branch remains a future ablation harness only.
- The current search still optimizes `final_int8_zlib_roundtrip val_bpb`, not train loss.
- Artifact size is still far below the `16 MB` cap, so static capacity remains the primary lever.

## Second Ada Scaling Wave

All runs below still used the old small-batch Ada command style:

- `TRAIN_SEQ_LEN=64`
- `TRAIN_BATCH_TOKENS=1024`
- implicit `GRAD_ACCUM_STEPS=8`

### Deeper and wider follow-up runs

| Variant | Params | Float `val_bpb` | Final int8 `val_bpb` | Total submission size int8+zlib |
|---|---:|---:|---:|---:|
| `8L 384d e288 4x2`, `MLP_MULT=3` | `3,977,125` | `2.3264` | `2.3283` | `3,336,685` bytes |
| `8L 448d e320 4x2`, `MLP_MULT=2` | `4,521,445` | `2.2990` | `2.3016` | `3,632,008` bytes |
| `10L 384d e288 5x2`, `MLP_MULT=2` | `3,834,284` | `2.3535` | `2.3580` | `3,298,760` bytes |

### Interpretation

1. Ada did eventually beat the final A5000 best line once the architecture was adapted to the new hardware regime.
2. The strongest result in the old small-batch Ada regime became:
   - `8L 448d e320 4x2`
   - final int8 roundtrip `2.3016`
3. `4x2` sharing is now a validated main-line lever, not just an A5000 rescue trick.
4. More depth by itself still does not solve the problem:
   - `10L 384d e288 5x2` lost clearly.

## First Ada Systems Modernization Sweep

After the architecture branch above won, the next question was whether the RTX 6000 Ada box was being severely underfed by the old T4-style training settings.

The first modernization sweep increased batch size and context length but still kept the training recipe tied to the old main-batch-derived meta sizing and fixed-step schedule assumptions.

### Naive modernization results

| Variant | Key system changes | Final int8 `val_bpb` |
|---|---|---:|
| `ada_sys_A` | `seq=64`, `batch=4096`, `accum=4` | `2.5258` |
| `ada_sys_B` | `seq=64`, `batch=4096`, `accum=2` | `2.5678` |
| `ada_sys_C` | `seq=128`, `batch=4096`, `accum=4` | `2.5218` |
| `ada_sys_D` | `seq=128`, `batch=8192`, `accum=4` | `2.4330` |

### Interpretation

1. The old small-batch regime was clearly underfeeding Ada.
2. Simply increasing system knobs without decoupling the rest of the recipe made results much worse.
3. The issue was not just throughput:
   - meta workload was still implicitly tied to local main-batch size
   - meta schedule was still effectively designed around fixed-step assumptions
4. `ada_sys_D` was the least bad of these runs, which still suggested that larger context and larger batch were the right direction.

## Script Changes Required For Real Ada Tuning

`train_gpt_competitive.py` was updated so the Ada search could be modernized without silently changing the optimization objective.

### New controls added

- `GRAD_ACCUM_STEPS`
- `TRAIN_META_PAIR_TOKENS`
- `TRAIN_META_START_TOKENS`
- `TRAIN_META_END_TOKENS`
- `TRAIN_META_START_MS`
- `TRAIN_META_END_MS`

### What these changes solved

1. Meta batch size can now be held fixed while the main training batch is scaled up.
2. Meta schedule can now be interpreted against wallclock progress, token progress, or explicit millisecond/token thresholds.
3. This makes 90-second calibration runs and 10-minute competition-style runs meaningful on Ada.

## Ada Calibration Sweep

After the decoupling patch, short wallclock-capped calibration runs were used to find a training regime that actually loads the RTX 6000 Ada box.

These runs are **not** direct replacements for the earlier 1500-step small-context experiments. Their purpose was to find the first credible Ada systems configuration.

Common calibration pattern:

- architecture: `8L 448d e320 4x2`
- `MAX_WALLCLOCK_SECONDS=90`
- `ITERATIONS=100000`
- `WARMUP_STEPS=20`
- lightweight validation/eval

### Calibration results

| Variant | Key system changes | Final int8 `val_bpb` | Peak allocated | Observed utilization |
|---|---|---:|---:|---|
| `ada_cal_A` | `seq=256`, `batch=16384`, `accum=2`, `meta_pair=2048` | `1.9567` | `1425 MiB` | `11%` VRAM, `42%` GPU |
| `ada_cal_B` | `seq=256`, `batch=32768`, `accum=2`, `meta_pair=2048` | `1.9028` | `2781 MiB` | `14%` VRAM, `50%` GPU |
| `ada_cal_C` | `seq=256`, `batch=16384`, `accum=1`, `meta_pair=2048` | `1.8722` | `2772 MiB` | `14%` VRAM, `33%` GPU |
| `ada_cal_D` | `seq=512`, `batch=16384`, `accum=2`, `meta_pair=4096` | `1.7676` | `1593 MiB` | `11%` VRAM, `41%` GPU |
| `ada_cal_E` | `seq=512`, `batch=16384`, `accum=1`, `meta_pair=4096` | `1.7132` | `2773 MiB` | `14%` VRAM, `52%` GPU |
| `ada_cal_F` | `seq=512`, `batch=32768`, `accum=1`, `meta_pair=4096` | `1.7026` | `5487 MiB` | `20%` VRAM, `27%` GPU |
| `ada_cal_G` | `seq=1024`, `batch=32768`, `accum=1`, `meta_pair=8192` | `1.6879` | `5489 MiB` | `20%` VRAM, `70%` GPU |

### Calibration conclusions

1. `GRAD_ACCUM_STEPS=1` is better than `2` on this box for the tested branch.
2. Increasing context length is a first-class Ada systems lever.
3. Bigger batch alone is not enough.
4. The first credible RTX 6000 Ada training regime is:
   - `TRAIN_SEQ_LEN=1024`
   - `TRAIN_BATCH_TOKENS=32768`
   - `GRAD_ACCUM_STEPS=1`
   - `TRAIN_META_PAIR_TOKENS=8192`
5. `ada_cal_G` is the first run that reached the practical stop rule for calibration:
   - stable
   - about `70%` GPU utilization
   - clearly best 90-second wallclock result

## Current Ada Baseline Going Forward

Architecture branch to continue from:

- `NUM_LAYERS=8`
- `NUM_UNIQUE_ATTN=4`
- `NUM_UNIQUE_MLP=2`
- `MODEL_DIM=448`
- `EMBED_DIM=320`
- `MLP_MULT=2`

Locked Ada systems configuration for real follow-up runs:

- `TRAIN_SEQ_LEN=1024`
- `TRAIN_BATCH_TOKENS=32768`
- `GRAD_ACCUM_STEPS=1`
- `TRAIN_META_PAIR_TOKENS=8192`
- `MAX_WALLCLOCK_SECONDS=600`
- `ITERATIONS=100000`
- `WARMUP_STEPS=20`
- `WARMDOWN_ITERS=0`

## Current State Summary

1. Final A5000 best line remains:
   - `6L 384d e288`, `MLP_MULT=3`
   - final int8 roundtrip `2.3290`
2. Best old-regime Ada architecture result is:
   - `8L 448d e320 4x2`
   - final int8 roundtrip `2.3016`
3. The first credible Ada systems regime is now locked:
   - `seq=1024`, `batch=32768`, `accum=1`, `meta_pair=8192`
4. The next Ada work should resume the main experiment program under that locked systems setup rather than continue low-utilization calibration.
