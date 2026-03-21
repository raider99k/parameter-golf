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
