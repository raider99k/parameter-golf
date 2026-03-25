# H100 Experiment Summary

Date: 2026-03-25

This file summarizes the main H100 experiments run on the current branch, with emphasis on `holy`, `competitive2`, and `competitive3`.

## Main Conclusions

- `holy` is currently the strongest original submission-shaped branch.
- Best known `holy` config so far:
  - `holy_2cores_xsa1_qat`
  - `final_int6_roundtrip_exact = 1.52901011`
- `competitive2` is not dead, but it is only viable in a narrow regime:
  - fast adapters on
  - meta on from the beginning
  - meta every step
  - moderate model size
  - partial sharing, not total sharing
- `competitive3` improved quality over `competitive2`, but current export size is far too large to be viable as a submission branch.

## Holy

### Baseline And Improvements

- `holy_true_1xh100_180s`
  - params: `10,511,087`
  - step_avg: about `113.93ms`
  - peak memory: about `30.4 GiB`
  - `DIAGNOSTIC post_ema val_bpb = 1.4636`
  - `final_int6_roundtrip_exact = 1.60324486`

- `holy_2cores_1xh100_180s`
  - params: `12,872,439`
  - `DIAGNOSTIC post_ema val_bpb = 1.4429`
  - `final_int6_roundtrip_exact = 1.55093235`

- `holy_2cores_qat_1xh100_180s`
  - earlier late-QAT
  - `DIAGNOSTIC post_ema val_bpb = 1.4430`
  - `final_int6_roundtrip_exact = 1.54466509`

- `holy_2cores_xsa1_qat_1xh100_180s`
  - current best known `holy` config
  - `XSA_LAST_N=1`
  - `NUM_SHARED_CORES=2`
  - `LATE_QAT_THRESHOLD=0.20`
  - `EMA_DECAY=0.997`
  - step_avg: about `110.68ms`
  - peak memory: about `28.2 GiB`
  - val at stop: `1.4368`
  - `DIAGNOSTIC post_ema val_bpb = 1.4394`
  - `final_int6_roundtrip_exact = 1.52901011`

### Holy Negative Tests

- `holy_2cores_xsa1_qat_embedfp16_1xh100_180s`
  - kept embedding tables in fp16 at export
  - `final_int6_roundtrip_exact = 1.52942212`
  - no useful improvement

- `holy_2cores_xsa1_qat_noema_1xh100_180s`
  - `EMA_DECAY=0`
  - `DIAGNOSTIC post_ema val_bpb = 1.4366`
  - `final_int6_roundtrip_exact = 1.57225662`
  - removing EMA made export worse

- `holy_2cores_xsa1_qat_tailint8_noema_1xh100_180s`
  - aligned exporter to QAT surrogate and forced `tail_blocks.*` to int8
  - `DIAGNOSTIC post_ema val_bpb = 1.4389`
  - `final_int6_roundtrip_exact = 1.58040253`
  - worse than baseline

### Holy Conclusions

- `holy` is real and strong.
- The branch is export-limited, but the simple “make export cleaner” fixes tested so far did not help.
- EMA is helping final export even when it slightly worsens live validation.
- The best `holy` regime so far is:
  - `NUM_SHARED_CORES=2`
  - `XSA_LAST_N=1`
  - earlier late-QAT
  - EMA enabled

## Competitive2

### Code Repairs

The following issues were identified and fixed:

- BitLinear export/QAT target mismatch
  - fixed so export reload, projection, and regularizer target the same latent object
- grouped ternary custom export packing bug
  - fixed `ternary_packed_group` handling in the custom binary format

These fixes were necessary before interpreting later `competitive2` results.

### Important Competitive2 Runs

- `competitive2_core_1xh100_180s_bestckpt`
  - params: `17,069,640`
  - step_avg: about `115.3ms`
  - peak memory: about `21.2 GiB`
  - `final_int8_zlib_roundtrip_exact = 2.12227630`

- `competitive2_repaired_1xh100_180s`
  - params: `6,186,541`
  - peak memory: about `24.3 GiB`
  - `final_int8_zlib_roundtrip_exact = 2.11625447`

- `competitive2_big_fastmeta_fulltime_1xh100_180s`
  - params: `6,186,541`
  - full-time meta from step 1
  - step_avg: about `173.99ms`
  - peak memory: about `24.3 GiB`
  - `final_int8_zlib_roundtrip_exact = 2.11045150`
  - best recent `competitive2` result

### Tiny Competitive2 Diagnostics

- `competitive2_tiny_fastmeta_1xh100_180s`
  - params: `2,287,519`
  - delayed meta
  - step_avg: about `101.81ms`
  - peak memory: about `16.4 GiB`
  - `final_int8_zlib_roundtrip_exact = 2.35265116`

- `competitive2_tiny_fastmeta_fulltime_1xh100_180s`
  - params: `2,287,519`
  - full-time meta from step 1
  - step_avg: about `133.10ms`
  - peak memory: about `16.4 GiB`
  - `final_int8_zlib_roundtrip_exact = 2.26395274`

- `competitive2_tiny_fastmeta_fulltime_b196k_1xh100_180s`
  - bigger batch
  - peak memory: about `24.6 GiB`
  - `final_int8_zlib_roundtrip_exact = 2.30588403`
  - bigger batch hurt

### Weird Competitive2 Test

- `competitive2_weird_universal_1xh100_180s`
  - params: `2,582,673`
  - `NUM_UNIQUE_ATTN=1`
  - `NUM_UNIQUE_MLP=1`
  - `MODEL_DIM=576`
  - `FAST_RANK=32`
  - step_avg: about `296.80ms`
  - peak memory: about `30.3 GiB`
  - `final_int8_zlib_roundtrip_exact = 2.34217917`

### Competitive2 Conclusions

- `competitive2` is not dead, but it only works at all in a narrow regime.
- Fast adapters and meta help the most when:
  - the model is small or medium
  - sharing is enabled
  - meta is on from step 1
  - meta runs every step
  - batch stays small enough to preserve update count
- Larger batch hurts.
- Very large models hurt under a `180s` budget because step cost destroys update count.
- Full universality (`NUM_UNIQUE_ATTN=1`, `NUM_UNIQUE_MLP=1`) is too extreme.
- Current best interpretation:
  - `competitive2` is a compact shared model whose main learning mechanism is continuous fast/meta adaptation
  - it is still far behind `holy`

## Competitive3

- `competitive3_core_1xh100_180s`
  - params: `27,005,532`
  - step_avg: about `148.87ms`
  - peak memory: about `28.2 GiB`
  - `final_int8_zlib_roundtrip_exact = 1.51549175`
  - total compressed submission size: about `30.55 MB`

### Competitive3 Conclusion

- `competitive3` improved quality substantially over `competitive2`
- but current export size is too large to be a viable submission branch
- it is better treated as a donor branch than as the current mainline

## Current Ranking Of Original Branches

1. `holy_2cores_xsa1_qat`
   - `1.52901011`
2. `competitive3_core`
   - better quality than `holy`, but not size-viable
3. `competitive2_big_fastmeta_fulltime`
   - `2.11045150`

## Recommended Mainline

- Keep `holy` as the main original branch
- Use this as the current reference config:
  - `NUM_SHARED_CORES=2`
  - `XSA_LAST_N=1`
  - `LATE_QAT_THRESHOLD=0.20`
  - `EMA_DECAY=0.997`

## Recommended Competitive2 Interpretation

- If continuing `competitive2`, treat it as:
  - a compact, partially shared, factorized model
  - with fast adapters on
  - with meta on from the beginning
  - with small batch
- Do not:
  - use giant models
  - use giant batches
  - rely on delayed meta
  - collapse uniqueness all the way to `1/1`
