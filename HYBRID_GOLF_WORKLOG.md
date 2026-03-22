# Hybrid Golf Worklog

This file records what we implemented in `hybrid_golf`, what we tested on the RTX 6000 Ada box, what broke, and what we currently believe.

It is intentionally operational rather than polished. The goal is to avoid losing project state across chat turns.

## Mission Context

We are **not** trying to win by nudging a weak proxy model a few basis points.

We are trying to reach a **submission-grade path** for Parameter Golf:

- under `16 MB` artifact budget
- under the `10 minute` training budget
- causal and leaderboard-safe
- competitive with the live frontier, not only the stale accepted table

As of **March 22, 2026**:

- official accepted README leader: `1.1428 BPB`
- live validated non-TTT frontier tracked in PR commentary: `1.1233 BPB` in `#414`
- live validated TTT frontier tracked in PR commentary: `1.1221 BPB` in `#398`, but under legality scrutiny

This means our internal results must be interpreted against the live frontier, not only the accepted leaderboard.

## Current Repo State

The historical run notes below are still useful, but they no longer fully describe what is implemented in `hybrid_golf`.

Current live code state:

- first-wave mainline acceleration is landed:
  - decimal `submission_total` accounting
  - factorized tied embeddings
  - attention / MLP bank sharing
  - NormFormer-lite
  - depth-aware initialization
  - Muon split optimizer
  - SWA-style averaging
  - projection-QAT
  - compression regularization
  - mixed-bit export (`mixed_v1`)

- second-wave acceleration is landed:
  - recurrent shared-stack depth
  - pass gates / pass modulation / pass q-gain
  - low-rank block deltas / logit delta
  - BitLinear / ternary latent weights
  - `mixed_v2` export for ternary latent tensors
  - config inheritance via `extends`
  - branch presets for:
    - Stage 0 root selection
    - recurrent branch
    - BitLinear branch
    - `doc_bias` parity overlays

- what is still not done:
  - automatic Stage 0 winner selection
  - automatic finalist advancement
  - final single-file submission condensation
  - repeat-run / significance harness

Interpretation:

- the toolkit is real and materially stronger than the historical proxy-era branch
- the next bottleneck is honest remote ranking of the landed presets, not another blind implementation spree

## What We Implemented

### Core toolkit

Implemented under `hybrid_golf/`:

- config loading and dotted overrides
- config inheritance via `extends`
- shard loading and byte-accounting-compatible tokenization
- GPT training and evaluation
- strict and exploratory policies
- `ngram`, `pointer`, and `doc_bias` experts
- writable-state adaptation / fast-weight path
- factorized tied embeddings
- attention / MLP bank sharing
- NormFormer-lite and depth-aware init
- Muon split optimizer
- SWA-style averaging
- projection-QAT and compression regularization
- recurrent shared-stack depth with pass controls
- low-rank recurrent deltas / logit delta
- BitLinear / ternary latent weights
- mixed-bit quantized export and roundtrip reload
- CLI entrypoints for train / eval / sweep

Main files:

- [hybrid_golf/config.py](c:\Users\pasqu\OpenChallenge\parameter-golf\hybrid_golf\config.py)
- [hybrid_golf/model.py](c:\Users\pasqu\OpenChallenge\parameter-golf\hybrid_golf\model.py)
- [hybrid_golf/train.py](c:\Users\pasqu\OpenChallenge\parameter-golf\hybrid_golf\train.py)
- [hybrid_golf/evaluate.py](c:\Users\pasqu\OpenChallenge\parameter-golf\hybrid_golf\evaluate.py)
- [hybrid_golf/policies.py](c:\Users\pasqu\OpenChallenge\parameter-golf\hybrid_golf\policies.py)
- [hybrid_golf/experts.py](c:\Users\pasqu\OpenChallenge\parameter-golf\hybrid_golf\experts.py)
- [hybrid_golf/export.py](c:\Users\pasqu\OpenChallenge\parameter-golf\hybrid_golf\export.py)
- [scripts/hg_train.py](c:\Users\pasqu\OpenChallenge\parameter-golf\scripts\hg_train.py)
- [scripts/hg_eval.py](c:\Users\pasqu\OpenChallenge\parameter-golf\scripts\hg_eval.py)
- [scripts/hg_sweep.py](c:\Users\pasqu\OpenChallenge\parameter-golf\scripts\hg_sweep.py)

### Presets and docs added

- [configs/hybrid/tiny_strict.json](c:\Users\pasqu\OpenChallenge\parameter-golf\configs\hybrid\tiny_strict.json)
- [configs/hybrid/tiny_exploratory.json](c:\Users\pasqu\OpenChallenge\parameter-golf\configs\hybrid\tiny_exploratory.json)
- [configs/hybrid/challenge_mainline_base.json](c:\Users\pasqu\OpenChallenge\parameter-golf\configs\hybrid\challenge_mainline_base.json)
- [configs/hybrid/challenge_mainline_shared.json](c:\Users\pasqu\OpenChallenge\parameter-golf\configs\hybrid\challenge_mainline_shared.json)
- [configs/hybrid/challenge_mainline_shared_qat.json](c:\Users\pasqu\OpenChallenge\parameter-golf\configs\hybrid\challenge_mainline_shared_qat.json)
- [configs/hybrid/challenge_mainline_recurrent.json](c:\Users\pasqu\OpenChallenge\parameter-golf\configs\hybrid\challenge_mainline_recurrent.json)
- [configs/hybrid/challenge_mainline_recurrent_delta.json](c:\Users\pasqu\OpenChallenge\parameter-golf\configs\hybrid\challenge_mainline_recurrent_delta.json)
- [configs/hybrid/challenge_mainline_bitlinear_mlp.json](c:\Users\pasqu\OpenChallenge\parameter-golf\configs\hybrid\challenge_mainline_bitlinear_mlp.json)
- [configs/hybrid/challenge_mainline_bitlinear_full.json](c:\Users\pasqu\OpenChallenge\parameter-golf\configs\hybrid\challenge_mainline_bitlinear_full.json)
- [configs/hybrid/challenge_parity.json](c:\Users\pasqu\OpenChallenge\parameter-golf\configs\hybrid\challenge_parity.json)
- [configs/hybrid/challenge_parity_lite.json](c:\Users\pasqu\OpenChallenge\parameter-golf\configs\hybrid\challenge_parity_lite.json)
- [configs/hybrid/challenge_parity_doc_bias.json](c:\Users\pasqu\OpenChallenge\parameter-golf\configs\hybrid\challenge_parity_doc_bias.json)
- [configs/hybrid/challenge_parity_lite_doc_bias.json](c:\Users\pasqu\OpenChallenge\parameter-golf\configs\hybrid\challenge_parity_lite_doc_bias.json)
- [ZERO_TO_SUBMISSION.md](c:\Users\pasqu\OpenChallenge\parameter-golf\ZERO_TO_SUBMISSION.md)
- [CHALLENGE_PARITY_CHECKLIST.md](c:\Users\pasqu\OpenChallenge\parameter-golf\CHALLENGE_PARITY_CHECKLIST.md)

### Important commits already pushed

- `36ad03b` Fix exploratory eval gradients and quiet eval output
- `f66b3bd` Use checkpoint config during eval
- `b1cc522` Add submission roadmap and export guardrails
- `22039c4` Add training-budget guardrail to roadmap
- `6c329e7` Add challenge parity eval preset and checklist
- `f698be5` Add challenge parity lite preset
- `bf09406` Add frontier reality check to roadmap
- `ad818ba` Fix eval preset overrides on checkpointed runs
- `ebb1d8e` Accelerate hybrid golf second wave

## Current protocol

The current mainline protocol is no longer:

- `strict + doc_bias` by default

The current mainline protocol is:

1. rank static priors with `doc_bias` off
2. use `challenge_parity_lite` for the broad ranking pass
3. use `challenge_parity` only on the top 1-2 candidates
4. re-test `doc_bias` exactly once on the single best static prior
5. keep `pointer`, `ngram`, `meta`, and adaptive extra pass frozen unless the new winner justifies reopening them

## Key bugs and fixes

### 1. Exploratory eval gradient crash

Symptom:

- exploratory eval crashed with:
  - `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`

Fix:

- patched adaptation / eval path
- committed in `36ad03b`

Outcome:

- exploratory eval ran successfully after patch

### 2. Eval shape mismatch when evaluating overridden training runs

Symptom:

- checkpoints trained with model overrides like `4x128` failed when evaluated with `tiny_strict.json`
- `hg_eval.py` rebuilt the default tiny model instead of the checkpoint model

Fix:

- `hg_eval.py` now uses the checkpoint's embedded config as the architecture base
- committed in `f66b3bd`

Outcome:

- checkpoints with overridden model dimensions evaluate correctly

### 3. Challenge parity presets were being silently ignored

Symptom:

- `challenge_parity.json` / `challenge_parity_lite.json` were intended to override eval behavior
- but `hg_eval.py` treated checkpoint config as total authority, so the preset file did not override eval/data/export sections

Fix:

- changed merge logic in [scripts/hg_eval.py](c:\Users\pasqu\OpenChallenge\parameter-golf\scripts\hg_eval.py)
- now:
  - checkpoint config supplies architecture and training shape
  - config file can still override eval/data/export behavior
- committed in `ad818ba`

Important consequence:

- any `challenge_parity_lite` numbers gathered **before** `ad818ba` should be treated as untrusted and rerun

### 4. Budget enforcement / export reporting was missing

Fix:

- added automatic export for serious runs
- added artifact byte reporting into train outputs
- added `export.fail_if_over_budget`
- committed in `b1cc522`

Outcome:

- serious runs now produce artifact stats and can hard-fail if over budget

## Test and validation work

### Local code checks

Repeatedly used:

- `python -m compileall ...`

This succeeded for the modified files when used.

### CLI smoke coverage

File:

- [tests/test_hybrid_cli_smoke.py](c:\Users\pasqu\OpenChallenge\parameter-golf\tests\test_hybrid_cli_smoke.py)

Purpose:

- trains a tiny checkpoint
- validates strict and exploratory CLI paths
- validates export presence
- now also validates that eval presets can override non-model behavior even when the checkpoint embeds a config

### Known local test issue

`pytest` on Windows is currently noisy because of ACL / temp-dir permission problems:

- `.pytest_tmp`
- `pytesttmp`
- `pytest-cache-files-*`
- temp directories under the Windows user temp path

This is an environment problem, not a discovered functional failure in the toolkit logic we changed. We worked around it where possible, but complete pytest confidence is still blocked by those ACL issues.

## Experimental phases

## Phase A: Tiny smoke runs

Goal:

- make sure train / strict eval / exploratory eval / export all work

Representative results:

- `tiny_strict`, strict eval on `32768` val tokens:
  - `val_bpb = 3.6722766988`
- `tiny_exploratory`, exploratory eval on `32768` val tokens:
  - `val_bpb = 3.6616555015`

Finding:

- exploratory was slightly better on the tiny smoke setup
- but this was not legal for submission and later turned out not to transfer

## Phase B: Early ablations on the tiny regime

Main ablations:

- base prior only
- experts only
- exploratory with/without prefix adaptation
- meta-enabled vs non-meta
- expert-by-expert ablations

Representative findings:

- base prior only on early tiny run:
  - `3.9581197730`
- experts only:
  - `3.6726499865`
- exploratory stronger prefix adaptation:
  - `3.6609931461`

But this regime was misleading.

What initially looked promising:

- `pointer`
- exploratory adaptation
- meta-training

What later happened:

- stronger proxy runs killed all of them

## Phase C: Stronger proxy regime with longer training and cleaner evaluator

We progressively moved to:

- strict policy
- no meta
- no pointer
- no ngram
- no adaptive extra pass
- doc_bias as the only surviving hybrid feature

Important intermediate results:

### `seq256` proxy

- `ada_proxy256_control300`, `doc_bias` only:
  - `3.2840634416`
- same checkpoint, base prior only:
  - `3.3149378850`

Finding:

- `doc_bias` helped
- `pointer` and `meta` hurt under stronger proxy conditions

### `seq512` fixed-step scaling

`4x128 seq512 2000`

- base: `2.5199993660`
- doc_bias: `2.5051932654`

`6x192 seq512 3000`

- base: `2.2898191576`
- doc_bias: `2.2744169147`

`8x256 seq512 4000`

- base: `2.1511664331`
- doc_bias: `2.1414859331`

`8x256 seq1024 4000`

- base: `2.1624576336`
- doc_bias: `2.1515104369`

Finding:

- `seq1024` lost to `seq512` in this regime
- capacity scaling helped more than context scaling

`10x320 seq512 5000`

- base: `2.0778587411`
- doc_bias: `2.0678622171`

`12x384 seq512 5000`

- base: `2.0685891538`
- doc_bias: `2.0586827029`

`14x448 seq512 5000`

- base: `2.0618325114`
- doc_bias: `2.0522689863`

Finding:

- fixed-step scaling kept helping a bit
- but we later stopped trusting fixed-step runs as submission-track comparisons

## Phase D: Wallclock-capped serious proxy runs

This was the first serious correction.

We switched from equal-step comparisons to equal-wallclock comparisons because the challenge budget is a hard `10 minute` constraint on `8xH100s`.

We used `540s` as a practical safety margin for serious proxy work.

### `14x448 @ 540s`

Train result:

- wallclock stop: `step=14737`
- artifact bytes: `12036875`
- within budget: `True`

Reduced-slice proxy eval (`256/512` windowing):

- base: `1.7833487392`
- doc_bias: `1.7800420820`

Finding:

- wallclock-capped training was much more important than earlier fixed-step conclusions suggested

### `12x384 @ 540s`

Train result:

- wallclock stop: `step=16628`
- artifact bytes: `7927472`
- within budget: `True`

Reduced-slice proxy eval (`256/512` windowing):

- base: `1.7774691114`
- doc_bias: `1.7747509822`

Finding:

- `12x384` beat `14x448` at equal wallclock
- it was also much smaller

### `10x320 @ 540s`

Train result:

- wallclock stop: `step=19567`
- artifact bytes: `4876437`
- within budget: `True`

Reduced-slice proxy eval (`256/512` windowing):

- base: `1.7741952247`
- doc_bias: `1.7728552068`

Finding:

- `10x320` slightly beat `12x384` at equal wallclock under the old reduced-slice proxy evaluator
- it was dramatically smaller

This made `10x320 @ 540s` the best **serious proxy** run before parity-lite reranking.

## Phase E: Challenge parity and parity-lite workflow

We added:

- [challenge_parity.json](c:\Users\pasqu\OpenChallenge\parameter-golf\configs\hybrid\challenge_parity.json)
- [challenge_parity_lite.json](c:\Users\pasqu\OpenChallenge\parameter-golf\configs\hybrid\challenge_parity_lite.json)
- [CHALLENGE_PARITY_CHECKLIST.md](c:\Users\pasqu\OpenChallenge\parameter-golf\CHALLENGE_PARITY_CHECKLIST.md)

Intended workflow:

1. use `challenge_parity_lite` for candidate ranking
2. use `challenge_parity` only for the best 1-2 runs
3. do not let proxy-only runs decide the final recipe

However:

- the first `challenge_parity_lite` results were gathered **before** the merge bug fix in `ad818ba`
- that means those results are not reliable enough to anchor final decisions

Recorded pre-fix numbers were:

`10x320`:

- parity-lite with doc_bias: `2.3464068470`
- parity-lite base: `2.3253763778`

`12x384`:

- parity-lite with doc_bias: `2.4031632979`
- parity-lite base: `2.3883804142`

What they appeared to suggest:

- `doc_bias` hurt under parity-lite
- `10x320` beat `12x384`

But the correct interpretation is:

- rerun parity-lite after `ad818ba` before trusting these as real

## Conclusions so far

### Confirmed

1. The toolkit is real.
   This is not a mock. Train/eval/export/roundtrip all work.

2. The early hybrid wins were mostly smoke-regime artifacts.

3. The current off-path branches are:

- `pointer`
- `ngram`
- `meta`
- adaptive extra pass

4. Equal-wallclock comparison matters far more than fixed-step comparison.

5. Capacity scaling helped more than context scaling in our proxy regime.

6. Export budget is not the current blocker.
   Even `14x448 @ 540s` exported under `16 MB`.

### No longer trusted

These earlier beliefs are no longer active:

- exploratory policy is a promising path
- pointer is part of the winning path
- meta-training is helping
- doc_bias is definitely good under parity-like evaluation

### Current working belief

The real bottleneck is the **static prior**, not the hybrid evaluator.

That is why we pivoted toward:

- challenge-aware training
- wallclock-capped comparisons
- parity / parity-lite gating
- frontier static-prior features from incumbent scripts

## Current best internal candidates

### Best serious proxy run under old reduced-slice proxy eval

`ada_proxy10x320_seq512_w540`

- base: `1.7741952247`
- doc_bias: `1.7728552068`
- artifact: `4,876,437` bytes

This is still the best confirmed serious-proxy run under the old `256/512` reduced-slice eval regime.

### Best fixed-step proxy run

`ada_strict14x448_seq512_5000`

- base: `2.0618325114`
- doc_bias: `2.0522689863`

This is no longer the main comparison anchor because fixed-step ranking is weaker than equal-wallclock ranking.

## Current roadmap implications

1. Stop treating the hybrid branch as the main source of gains.
2. Stop reopening `pointer`, `ngram`, `meta`, and exploratory eval without new evidence.
3. Keep budget-aware training and export reporting on every serious run.
4. Re-run `challenge_parity_lite` comparisons **after** `ad818ba` before using them as a real ranking gate.
5. Shift implementation effort toward frontier static-prior features:
   - optimizer split / Muon-style matrix updates
   - EMA / Tight SWA style averaging
   - export-aware training / QAT
   - stronger compact prior design

## Frontier-aligned interpretation

The live frontier is around `1.1233` non-TTT.

Our best serious proxy run is still far away.

So the correct reading is:

- we have a functioning research harness
- we have removed several dead-end branches
- we have learned how to compare runs more honestly
- but we are still not close to the live frontier

The project should now optimize for:

- stronger compact static priors
- legal causal evaluation
- fast candidate triage with parity-lite
- full parity only for top candidates

## Immediate next steps

1. Re-run `10x320` and `12x384` with [challenge_parity_lite.json](c:\Users\pasqu\OpenChallenge\parameter-golf\configs\hybrid\challenge_parity_lite.json) **after** the `ad818ba` fix and record the new valid ranking.
2. Freeze the winning static prior family from that rerun.
3. Implement frontier static-prior features instead of more hybrid expert work.
4. Use [challenge_parity.json](c:\Users\pasqu\OpenChallenge\parameter-golf\configs\hybrid\challenge_parity.json) only on the top 1-2 candidates.
