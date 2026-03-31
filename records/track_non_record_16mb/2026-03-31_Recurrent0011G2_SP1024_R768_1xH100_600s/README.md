This record captures a non-record submission for the stripped recurrent `0011 + g2` candidate.

Author metadata prepared from the local environment:
- Author: `Pasquale`
- GitHub ID: `raider99k`

This run is intended for the unlimited-compute / non-record track, not the main 10-minute leaderboard. It was evaluated on `1xH100` with `MAX_WALLCLOCK_SECONDS=600`, so it is not the official `8xH100` leaderboard regime.

Approach:
- This submission packages a stripped recurrent trainer around the strongest branch found in local search: a fixed four-step `block_anchor` solver with schedule `0,0,1,1` (`A-A-C-C`) and grouped anchor gating (`g2`).
- The core idea is to make phase structure explicit while keeping the exported artifact simple: two shared proposal-style passes followed by two shared corrective passes, with no virtual layers, tails, slots, dual-state memory, or other experimental branches in the submitted code snapshot.
- The submitted `train_gpt.py` is intentionally narrow so that the counted code focuses on the actual candidate rather than the broader architecture-search harness.

Trainer snapshot:
- Source run used the repo-root `train_gpt_candidate.py` from `dev-branch`
- Record-local `train_gpt.py` is an exact copy of that stripped candidate trainer for submission packaging
- Code size: `48881` bytes
- Line count: `1086`

Configuration:
- Track: `non-record-unlimited-compute-16mb`
- Family: recurrent-only, fixed `block_anchor`
- Recurrence: `REC_CORE_SCHEDULE=0,0,1,1`
- Grouped anchor gates: `REC_COEFF_GROUPS=2`
- Layout: `MODEL_DIM=768 NUM_HEADS=12 NUM_KV_HEADS=6 NUM_SHARED_CORES=2`
- Corrective path: `CORRECTIVE_CORE_INDEX=1 CORRECTIVE_CONTROL_MODE=shared`
- Disabled features: virtual layers, slots, corrective tails, C2 refiner
- Data/tokenizer: published `fineweb10B_sp1024` / `fineweb_1024_bpe.model`

Command used for the run:
- Run from the repository root so the relative `DATA_PATH` and `TOKENIZER_PATH` stay valid.
```bash
RUN_ID=gpt_candidate_r768_0011_g2_1xh100_600s \
MODEL_FAMILY=recurrent \
MODEL_DIM=768 \
NUM_HEADS=12 \
NUM_KV_HEADS=6 \
NUM_SHARED_CORES=2 \
REC_STEPS_START=4 \
REC_STEPS_MAX=4 \
REC_SOLVER=block_anchor \
REC_CORE_SCHEDULE=0,0,1,1 \
REC_COEFF_GROUPS=2 \
CORRECTIVE_CORE_INDEX=1 \
CORRECTIVE_TAIL_TYPE=none \
CORRECTIVE_CONTROL_MODE=shared \
CORRECTIVE_C2_MODE=none \
START_CORE_INDEX=0 \
ALTERNATE_START_CORE=0 \
USE_VIRTUAL_LAYERS=0 \
USE_PRECOMPOSE=1 \
SLOT_ENABLED=0 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
SEED=3407 \
TRAIN_BATCH_TOKENS=131072 \
TRAIN_SEQ_LEN=2048 \
GRAD_ACCUM_STEPS=1 \
MAX_WALLCLOCK_SECONDS=600 \
WARMUP_STEPS=5 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=524288 \
VAL_TOKENS_LIMIT=65537 \
TRAIN_LOG_EVERY=100 \
python train_gpt_candidate.py
```

Key metrics (from `train.log`):
- Timed training stopped at `7432/20000` steps due to the wallclock cap.
- Pre-quant eval at stop: `val_loss:2.2774`, `val_bpb:1.3465`
- Post-quant roundtrip eval: `val_loss:2.2929`, `val_bpb:1.3557`
- Exact printed metric: `final_int8_zlib_roundtrip_exact val_bpb:1.35568780`
- Train time: `600151ms` (`step_avg:80.75ms`)
- Peak memory: `17417 MiB allocated`, `17572 MiB reserved`
- Model params: `13767194`
- Serialized model int8+zlib: `12569851 bytes`
- Code size: `48881 bytes`
- Total submission size int8+zlib: `12618732 bytes`

Training volume:
- Global batch: `131072` tokens/step
- Total train tokens seen: `974127104`

Included files:
- `train_gpt.py` - exact code snapshot intended for submission packaging
- `train.log` - exact training log for the run reported above
- `submission.json` - metadata for the non-record submission
