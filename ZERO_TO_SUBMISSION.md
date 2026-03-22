# Zero To Submission

## Mission

Build a **legal parameter-golf submission** that is competitive under the actual challenge constraints:

- artifact size must fit within **16 MB**
- training must fit within the **10 minute challenge budget**
- evaluation must be **causal and leaderboard-safe**
- results must matter under the real objective, not just under toy smoke runs
- the final system should be a **submission candidate**, not only a research sandbox

This document is the operating plan from the current toolkit state to a real submission.

## Current Position

We already have a working research toolkit under `hybrid_golf/`:

- train / eval / sweep CLIs work
- strict and exploratory policies work
- export and roundtrip work
- shard loading and byte accounting work
- online experts work
- writable-state adaptation exists

This is **not a mock**.

But it is also **not yet a challenge-faithful submission stack**.

What is currently true:

- the best surviving evaluator idea so far is `strict + doc_bias`
- `pointer`, `ngram`, `meta`, and adaptive extra pass are currently off the winning path
- capacity scaling is helping more than context scaling in the current proxy regime
- we are **not yet enforcing the submission budget on every run**
- we are **not yet running a frontier-strength static prior**
- export is currently **int8 + zlib**, not a frontier-grade mixed-bit submission export

## Answer To The Key Question

### Is it okay that we are testing without all the frontier implementations?

Yes, but only for the **right purpose**.

It is okay when the goal is:

- proving that the toolkit is correct
- killing broken ideas quickly
- checking whether a hybrid idea has any visible signal at all
- measuring rough runtime and export behavior

It is **not okay** when the goal is:

- claiming we are close to a real submission
- comparing ourselves seriously to the live frontier
- deciding that a hybrid method survives on top of a strong static prior

Reason:

Without a strong frontier-like static prior, we may be measuring gains that disappear once the base LM becomes competitive. A weak model can make simplistic online tricks look better than they really are.

So the correct workflow is:

1. use simplified runs for engineering and pruning
2. build a challenge-faithful baseline
3. only then judge whether hybrid additions are worth keeping

## Non-Negotiable Constraints

Every serious branch must eventually satisfy all of these:

1. **Artifact budget**
   The final exported artifact must be under `16 MB`.

2. **Training budget**
   The final training recipe must fit within the challenge's `10 minute` budget.

3. **Legal evaluation**
   Default path must stay within `strict_causal`.

4. **Real byte accounting**
   BPB must always be computed with the challenge-aligned byte metric.

5. **Submission realism**
   Candidate decisions must be made using runs that are close enough to the real regime to transfer.

6. **Reproducibility**
   Each serious run must leave behind:
   - resolved config
   - checkpoint
   - export size
   - train wallclock
   - eval result
   - notes on whether the result is smoke, proxy, or submission-grade

## What We Have Implemented

### Real and usable now

- config loader with overrides
- challenge shard reader
- tokenizer / byte-accounting layer
- GPT model training and evaluation
- strict and exploratory evaluator policies
- ngram / pointer / doc_bias experts
- writable-state adaptation
- roundtrip export / reload
- checkpoint-aware eval config loading

### Implemented but still simplified

- export:
  current export is int8-centric and compressed with zlib
- expert gate:
  heuristic and hand-tuned, not learned or calibrated
- meta-training:
  present, but currently not on the winning path
- recurrent adaptive compute:
  present, but currently not on the winning path

### Not implemented yet from the bigger submission vision

- mixed-bit frontier export path (`int6` / `int5` / more selective precision allocation)
- QAT / late-QAT
- EMA / Tight SWA
- stronger frontier architectural stack like partial RoPE / LN scaling / XSA-like additions
- automatic artifact-budget enforcement on every serious run
- systematic significance / repeat-run harness
- full submission condensation path into a final minimal artifact script

## Incumbent Script Takeaways

Two existing scripts contain ideas that are more submission-relevant than our current proxy-only branch:

- [train_gpt_competitive.py](/c:/Users/pasqu/OpenChallenge/parameter-golf/train_gpt_competitive.py)
- [train_gpt_extreme.py](/c:/Users/pasqu/OpenChallenge/parameter-golf/train_gpt_extreme.py)

These scripts matter because they already encode a more challenge-faithful philosophy than the current `hybrid_golf` baseline: stronger static priors, export-aware training, and compute-for-bytes architectural tradeoffs.

### Competitive script ideas worth pulling into the roadmap

Highest value:

- **export-aware training**
  - projection-to-export-grid QAT
  - compression regularizer
  - explicit roundtrip validation and submission-size logging

- **better export discipline**
  - selective keep-float policy
  - per-row scaling
  - exact treatment of small / control tensors

- **optimizer split with Muon**
  - matrix params on Muon
  - embeddings / head / scalars on Adam

- **factorized tied embeddings**
  - directly relevant to budget-aware capacity

- **parameter sharing through attention / MLP banks**
  - a real bytes-vs-compute lever

Important but bigger branch:

- **BitLinear / ternary latent weights**
  - highly relevant to submission compression
  - more invasive than the items above, so it should be treated as a dedicated branch

Lower priority for the submission path right now:

- fast adapters / meta-shaping
  - they exist in the script, but our current results do not justify them as an active baseline branch

### Extreme script ideas worth pulling into the roadmap

Highest value:

- **shared-stack recurrence**
  - reuse the same stack for multiple passes with learned pass gates
  - this is exactly the kind of compute-for-bytes trade the contest rewards

- **depth-aware residual / skip initialization**
  - cheap architectural stabilization for deeper effective stacks

- **NormFormer-lite branch post-normalization**
  - small architectural change with plausible optimization benefits

Medium priority:

- **layer / pass modulation**
  - potentially useful once the recurrent/shared branch exists

- **low-rank block deltas / logit delta**
  - plausible low-byte refinement knobs, but not clearly ahead of stronger export work

What the extreme script explicitly confirms:

- aggressive architecture work is in scope
- eval-time TTT is not on the mainline path
- the more submission-relevant frontier is a stronger compact prior, not a looser evaluator

## Prioritized Frontier Backlog

Ordered by submission relevance, not by novelty:

1. **Automatic budget accounting**
   Every serious run must produce export bytes and roundtrip metrics.

2. **Export-aware static prior**
   Bring over the competitive-script ideas:
   - projection-QAT
   - compression regularization
   - selective precision retention
   - stronger export accounting

3. **Optimizer and embedding improvements**
   Bring over:
   - Muon optimizer split
   - factorized tied embeddings

4. **Compute-for-bytes backbone branch**
   Explore one serious branch based on:
   - partial attention / MLP sharing
   - or recurrent shared-stack depth

5. **Advanced compression branch**
   Explore:
   - BitLinear / ternary latent weights
   - more aggressive mixed-precision export

6. **Hybrid additions on top**
   Only after the stronger prior exists:
   - re-test `doc_bias`
   - then optionally revisit `pointer` or writable-state methods

## Strategic Principle

We should stop asking one question:

> "Can we keep scaling the current proxy model?"

And ask the correct one:

> "What is the shortest path to a legal, under-16MB, frontier-relevant submission candidate?"

That changes the roadmap.

## Roadmap

## Phase 0: Freeze The Current Lessons

Status: mostly done

What we learned so far:

- `doc_bias` is the only hybrid addition with stable positive signal so far
- `pointer`, `ngram`, `meta`, and adaptive extra pass should be treated as inactive branches
- `seq_len=1024` was worse than `seq_len=512` in the current proxy regime
- scaling model capacity helped more than adding the other online features

Decision:

- freeze the current clean evaluator baseline as:
  - `strict_causal`
  - `doc_bias` enabled
  - `ngram` disabled
  - `pointer` disabled
  - `meta` disabled
  - `adaptive_extra_pass` disabled

This is the default research baseline until something clearly beats it.

## Phase 1: Make Every Serious Run Challenge-Aware

Status: not complete

Before scaling further, add the following:

1. automatic export after every serious train run
2. artifact-byte reporting in train/eval summaries
3. an explicit warning or failure when export size exceeds a configured budget target
4. run labels such as:
   - `smoke`
   - `proxy`
   - `submission_candidate`

Why:

Right now we are finding useful trends, but we are still under-enforcing the main contest constraint. That is dangerous.

Deliverable:

- every serious run should produce:
  - checkpoint
  - resolved config
  - exported artifact
  - artifact bytes
  - measured train wallclock
  - eval result

Wallclock rule:

- fixed-step runs are acceptable for `smoke` debugging
- fixed-step runs are acceptable for rough `proxy` ranking only when they are clearly labeled as such
- any run that influences submission decisions must be compared under an explicit wallclock cap, not only equal step counts

## Phase 2: Build A Challenge-Faithful Static Prior

Status: not complete

This is the biggest missing piece.

We should not keep treating the current strict baseline as if it were frontier-like. It is a clean proxy, not yet a serious static prior.

Goal:

Implement enough of the modern frontier stack to get a **strong compact prior** under the 16 MB export budget.

Priority order:

1. stronger export path
   - mixed precision / mixed bit-width export
   - better compression than current int8+zlib baseline
   - competitive-script style selective keep-float and projection-aware export

2. training-time robustness to export
   - QAT or at least export-aware finetuning
   - projection-QAT and compression regularization are the first concrete candidates

3. budget-aware capacity search
   - measure quality as a function of exported bytes, not only model dimensions
   - include factorized tied embeddings and optimizer-split variants

4. compute-for-bytes architectural branch
   - partial attention / MLP sharing
   - or recurrent shared-stack depth from the extreme script

5. selected frontier features only if they are byte-efficient
   - not "implement everything"
   - prioritize depth-aware residual init, NormFormer-lite, and other cheap stabilizers before heavier branches

Meaning:

We do **not** need every frontier trick before doing research.
But we **do** need a strong enough static prior that hybrid gains are measured on top of something realistic.

## Phase 3: Re-test Hybrid Additions On Top Of The Strong Prior

Status: blocked on Phase 2

Only after the stronger prior exists should we revisit hybrid components.

Order:

1. `doc_bias`
   - confirm it still buys residual BPB on top of a stronger static prior

2. `pointer`
   - re-test only if there is reason to think local repetition still leaves useful entropy

3. `ngram`
   - low priority unless retuned and clearly motivated

4. writable-state adaptation
   - only if it helps under `strict_causal`
   - exploratory-only gains do not count as progress toward submission

Decision gate:

If an addition does not improve the stronger prior by a visible and stable margin, it is cut.

## Phase 4: Submission Candidate Construction

Status: not started

A run can only be called a submission candidate if all of this is true:

1. exported artifact is under `16 MB`
2. training recipe fits within the `10 minute` budget
3. evaluation path is `strict_causal`
4. roundtrip score matches closely
5. result is reproduced on a meaningful validation slice
6. the candidate beats the previous internal baseline

Expected output:

- one named config that is considered the current best candidate
- one exported artifact
- one reproducible evaluation command

## Phase 5: Hardening And Final Condensation

Status: not started

Once a submission candidate exists:

1. simplify the stack where possible
2. remove dead research toggles from the final path
3. document exact run commands
4. minimize evaluation ambiguity
5. optionally condense the final recipe into a simpler submission-oriented path if needed

## What We Should Stop Doing

Until Phase 1 and Phase 2 are tighter, stop doing these by default:

- scaling capacity without checking export size
- treating exploratory-policy gains as meaningful submission progress
- reopening `pointer`, `ngram`, or `meta` without new evidence
- claiming a branch is promising only because it wins on tiny smoke settings
- drifting into "add more tricks" mode without budget discipline

## Immediate Next Actions

These are the highest-priority concrete tasks.

1. Add automatic artifact export and byte reporting to serious runs.
2. Make wallclock a first-class metric for serious runs.
3. Define what counts as a `serious run` in config terms.
4. Freeze the current clean baseline:
   - strict
   - doc_bias only
   - no pointer
   - no ngram
   - no meta
   - no adaptive extra pass
5. Start the budget-aware static-prior phase instead of blind capacity scaling.
6. Re-test hybrid additions only after the stronger prior exists.

## Current Baseline To Beat

Internal research baseline right now:

- model family: strict baseline branch
- evaluator:
  - `strict_causal`
  - `doc_bias` enabled
  - `ngram` disabled
  - `pointer` disabled
  - `meta` disabled
  - `adaptive_extra_pass` disabled

Best observed proxy result so far:

- `ada_strict12x384_seq512_5000`
- `val_bpb = 2.0586827028568235`
- validation slice: `131072` tokens

This is the current internal number to beat, but it is still a **proxy baseline**, not yet a final submission-grade result.

## Final Rule

From this point on, every experiment should answer one of only three questions:

1. Does it improve the **budget-aware static prior**?
2. Does it improve the **strict legal evaluator** on top of that prior?
3. Does it improve the **submission artifact** without breaking accuracy?

If it answers none of those, it is not on the critical path to submission.

Additional rule for experiment hygiene:

- if a run is not wallclock-capped, it is not allowed to decide the final submission recipe by itself

## Challenge-Parity Eval

The repository now includes:

- [challenge_parity.json](/c:/Users/pasqu/OpenChallenge/parameter-golf/configs/hybrid/challenge_parity.json)
- [CHALLENGE_PARITY_CHECKLIST.md](/c:/Users/pasqu/OpenChallenge/parameter-golf/CHALLENGE_PARITY_CHECKLIST.md)

These define the minimum standard for treating a `hybrid_golf` result as challenge-ready rather than merely proxy-valid.
