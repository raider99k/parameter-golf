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

## Protocol Reset: Trust Wallclock-Tracked Runs

After the first Ada calibration and architecture sweeps, it became clear that some conclusions from the mixed fixed-step / small-batch / changed-systems regime should be treated as provisional.

The key issue was simple:

- once throughput and context changed materially, comparing final states from fixed-step runs stopped being a clean proxy for the `10min` competition objective
- best performance could occur well before the final checkpoint

To fix this, `train_gpt_competitive.py` was updated again to support:

- `VAL_LOSS_EVERY_SECONDS`
- `RESTORE_BEST_VAL_CHECKPOINT`
- `best_val_update`
- `best_val_summary`

This changed the evaluation methodology:

1. use fixed wallclock budgets for search
2. validate at fixed wallclock intervals
3. restore the best validation checkpoint before final roundtrip evaluation

From this point onward, the most trustworthy Ada evidence comes from the locked wallclock-tracked regime, not from the earlier mixed protocol.

## 180s Wallclock Re-Baseline

All runs below used the same locked Ada systems config:

- `TRAIN_SEQ_LEN=1024`
- `TRAIN_BATCH_TOKENS=32768`
- `GRAD_ACCUM_STEPS=1`
- `TRAIN_META_PAIR_TOKENS=8192`
- `MAX_WALLCLOCK_SECONDS=180`
- `VAL_LOSS_EVERY_SECONDS=90`
- `RESTORE_BEST_VAL_CHECKPOINT=1`

### First clean 180s wallclock baseline

| Variant | Params | Best `val_bpb` | Final int8 `val_bpb` | Total submission size int8+zlib |
|---|---:|---:|---:|---:|
| `8L 448d e320 4x2`, `MLP_MULT=2` | `4,521,445` | `1.8401` | `1.8438` | `3,639,868` bytes |
| `8L 448d e320 4x2`, `MLP_MULT=3` | `5,326,053` | `1.8325` | `1.8356` | `4,454,750` bytes |
| `8L 512d e384 4x2`, `MLP_MULT=2` | `5,874,213` | `1.8011` | `1.8022` | `4,663,512` bytes |

### Interpretation

1. Under a clean 180-second wallclock comparison, width was still the main lever.
2. `MLP_MULT=3` helped at `448d`, but not enough to beat a wider `512d` model.
3. This established the first clean wallclock-tracked main branch:
   - `8L 512d e384 4x2`

## Aggressive Ada Scaling Wave

With the wallclock protocol locked, the search shifted to aggressive geometric jumps rather than conservative local nudges.

### First aggressive wave

| Variant | Params | Best `val_bpb` | Final int8 `val_bpb` | Total submission size int8+zlib |
|---|---:|---:|---:|---:|
| `8L 640d e512 4x2`, `MLP_MULT=2` | `9,095,845` | `1.7921` | `1.7942` | `7,091,031` bytes |
| `10L 640d e512 5x2`, `MLP_MULT=2` | `10,334,892` | `1.8290` | `1.8317` | `8,535,631` bytes |
| `8L 768d e640 4x2`, `MLP_MULT=2` | `13,005,605` | `1.7999` | `1.8016` | `10,021,860` bytes |

### Interpretation

1. Width still paid at least one more large step:
   - `8L 640d e512 4x2` became the new leader.
2. Depth continued to lose even after moving into the new wallclock-tracked regime:
   - `10L 640d e512 5x2` was clearly worse.
3. `768d` did not beat `640d`, which suggested that under a fixed `180s` budget, very large width was already running into wallclock-efficiency limits.

## Ridge Search Around 640d

After `640d` won, the search moved to the local ridge around that width while staying aggressive.

### Follow-up runs

| Variant | Params | Best `val_bpb` | Final int8 `val_bpb` | Total submission size int8+zlib |
|---|---:|---:|---:|---:|
| `8L 640d e512 4x2`, `MLP_MULT=3` | `10,736,805` | `1.7816` | `1.7832` | `8,743,097` bytes |
| `8L 704d e576 4x2`, `MLP_MULT=2` | `10,964,709` | `1.8098` | `1.8102` | `8,493,112` bytes |
| `8L 704d e576 4x2`, `MLP_MULT=3` | `12,949,989` | `1.8500` | `1.8500` | `10,485,195` bytes |
| `8L 768d e640 4x2`, `MLP_MULT=3` | `15,367,973` | `1.7848` | `1.7872` | `12,390,285` bytes |
| `8L 896d e704 4x2`, `MLP_MULT=2` | `17,480,613` | `1.8778` | `1.8808` | `13,341,459` bytes |
| `8L 896d e704 4x2`, `MLP_MULT=3` | `20,695,461` | `1.9099` | `1.9123` | `16,562,753` bytes |

### Interpretation

1. `MLP_MULT=3` is now fully validated on the main Ada branch:
   - `8L 640d e512 4x2 mlp3` is the best result so far.
2. The local optimum under the current `180s` wallclock budget sits around `640d`, not `704d`, `768d`, or `896d`.
3. Going past `640d` increased size and reduced wallclock efficiency without improving the metric enough.
4. `8L 896d e704 4x2 mlp3` exceeded the `16MB` artifact cap and is not a valid submission direction.
5. The depth branch should remain paused unless a later regime change creates new evidence.

## Best 180s Ada Line

Under the current trusted wallclock-tracked methodology at `180s`, the best line was:

- `NUM_LAYERS=8`
- `NUM_UNIQUE_ATTN=4`
- `NUM_UNIQUE_MLP=2`
- `MODEL_DIM=640`
- `EMBED_DIM=512`
- `MLP_MULT=3`
- `TRAIN_SEQ_LEN=1024`
- `TRAIN_BATCH_TOKENS=32768`
- `GRAD_ACCUM_STEPS=1`
- `TRAIN_META_PAIR_TOKENS=8192`
- `MAX_WALLCLOCK_SECONDS=180`
- `VAL_LOSS_EVERY_SECONDS=90`
- `RESTORE_BEST_VAL_CHECKPOINT=1`

Best observed result on `SEED=3407`:

- params: `10,736,805`
- best `val_bpb`: `1.7816`
- final int8 roundtrip `val_bpb`: `1.7832`
- total submission size int8+zlib: `8,743,097` bytes

## State Summary After 180s Ridge Search

1. Older fixed-step / early-Ada results remain useful as exploration history, but they are no longer the strongest evidence base.
2. The trusted current methodology is the wallclock-tracked Ada regime with best-checkpoint restore.
3. The best current Ada line is:
   - `8L 640d e512 4x2 mlp3`
4. Width remains the dominant scaling lever.
5. Depth is still losing under the current budget.
6. There is still artifact-size headroom below `16MB`, but the current `180s` budget already shows diminishing returns past `640d`.

## Re-Exploring Moonshot and Extreme on Ada

Once the search had stabilized around `8L 640d e512 4x2 mlp3`, the previously discarded `moonshot` and `extreme` branches were re-opened under the stronger Ada systems regime.

The question was no longer whether those branches looked good under early small-GPU settings. The question became:

- can `moonshot` full meta-learning beat the current competitive line when the outer shape is held fixed?
- can `extreme` architecture changes beat the same-shape competitive incumbent?

### 180s branch re-checks

| Variant | Params | Best `val_bpb` | Final int8 `val_bpb` | Total submission size int8+zlib |
|---|---:|---:|---:|---:|
| `moonshot` native default shape | `17,069,640` | `1.8483` | `1.8483` | `7,323,709` bytes |
| `moonshot` default full document meta | `17,113,217` | `1.8210` | `1.8211` | `7,494,182` bytes |
| `moonshot` same shape `8L 640d e512 4x2 mlp3`, full meta | `10,736,805` | `1.7946` | `1.7972` | `8,717,493` bytes |
| `extreme` same shape `8L 640d e512 4x2 mlp3`, full branch | `10,768,165` | `1.8430` | `1.8452` | `8,788,566` bytes |

### Interpretation

1. `moonshot` native/default settings remained clearly behind the competitive line.
2. Full `moonshot` document-meta training helped `moonshot` materially.
3. The only `moonshot` variant worth taking seriously was the same-shape full-meta run.
4. `extreme` was not competitive even when matched to the same outer shape.

## 300s Head-to-Head: Competitive vs Moonshot Full Meta

The next question was whether the same-shape `moonshot` full-meta branch would catch up at a longer wallclock budget.

### 300s comparison

| Variant | Params | Best `val_bpb` | Final int8 `val_bpb` | Total submission size int8+zlib |
|---|---:|---:|---:|---:|
| `competitive` same shape `8L 640d e512 4x2 mlp3`, fast/meta on | `10,736,805` | `1.7375` | `1.7403` | `8,770,099` bytes |
| `moonshot` same shape full-meta document `replace` | `10,736,805` | `1.7508` | `1.7529` | `8,751,439` bytes |

### Failed auxiliary full-meta attempt

An additional `moonshot` same-shape `aux` document-meta run at `300s` was attempted, but it is **not** a valid comparison result:

- the first `90s` validation (`1.8991`) happened before meta became active
- the run hit OOM when the auxiliary full-meta path became active
- the GPU was shared with another large process at the time

This established that `aux` document-meta is much harder to run fairly at the full same-shape settings.

### Fit probe for auxiliary document-meta

To test feasibility, a reduced `aux` document-meta probe was run with:

- `TRAIN_BATCH_TOKENS=16384`
- `TRAIN_META_BATCH_DOCS=4`
- `TRAIN_META_ADAPT_TOKENS=2048`
- `TRAIN_META_QUERY_TOKENS=1024`

Result:

- params: `10,736,805`
- best `val_bpb`: `1.7916`
- final int8 roundtrip `val_bpb`: `1.7925`
- total submission size int8+zlib: `8,769,347` bytes

This showed that auxiliary document-meta can fit if the workload is reduced, but that result is not apples-to-apples against the main line.

### Interpretation

1. Same-shape `moonshot` full-meta `replace` is viable on Ada.
2. It still lost to the competitive branch at `300s`:
   - `1.7529` vs `1.7403`
3. `moonshot` full meta-learning remained interesting, but no longer looked like the best use of search budget.
4. `extreme` remained a losing branch and should stay deprioritized.

## Fast Adapter Ablation on the Main Line

The decisive follow-up test was to keep the same winning outer shape and simply disable fast adapters / meta entirely.

### 300s fast-adapter ablation

| Variant | Params | Best `val_bpb` | Final int8 `val_bpb` | Total submission size int8+zlib |
|---|---:|---:|---:|---:|
| `competitive` same shape `8L 640d e512 4x2 mlp3`, fast/meta on | `10,736,805` | `1.7375` | `1.7403` | `8,770,099` bytes |
| `competitive` same shape `8L 640d e512 4x2 mlp3`, `USE_FAST_ADAPTERS=0` | `10,716,304` | `1.6892` | `1.6902` | `8,627,389` bytes |

### Interpretation

1. Under the stronger Ada regime and a `300s` budget, the fast-adapter/meta machinery is not helping the current main line.
2. Disabling fast adapters improved the metric substantially:
   - `1.6902` vs `1.7403`
3. The no-fast run was also:
   - slightly smaller
   - faster per step
   - much cheaper than `moonshot` full meta in VRAM
4. This superseded the earlier `180s` conclusion that the fast/meta-enabled line was the current best.

## Current Best Ada Line

Under the current strongest evidence base, the best line is now:

- `NUM_LAYERS=8`
- `NUM_UNIQUE_ATTN=4`
- `NUM_UNIQUE_MLP=2`
- `MODEL_DIM=640`
- `EMBED_DIM=512`
- `MLP_MULT=3`
- `TRAIN_SEQ_LEN=1024`
- `TRAIN_BATCH_TOKENS=32768`
- `GRAD_ACCUM_STEPS=1`
- `USE_FAST_ADAPTERS=0`
- `MAX_WALLCLOCK_SECONDS=300`
- `VAL_LOSS_EVERY_SECONDS=90`
- `RESTORE_BEST_VAL_CHECKPOINT=1`

Best observed result on `SEED=3407`:

- params: `10,716,304`
- best `val_bpb`: `1.6892`
- final int8 roundtrip `val_bpb`: `1.6902`
- total submission size int8+zlib: `8,627,389` bytes

## Latest State Summary

1. The strongest current line is still the same outer architecture family:
   - `8L 640d e512 4x2 mlp3`
2. But the best version of that line is now the plain supervised one:
   - `USE_FAST_ADAPTERS=0`
3. `moonshot` full meta-learning is real and competitive, but still behind the best competitive no-fast run.
4. `extreme` is not competitive under matched-shape Ada testing.
5. The next most justified work is on dormant competitive features (`compile`, `QAT`, compression-aware training), not more moonshot/extreme scaling.
