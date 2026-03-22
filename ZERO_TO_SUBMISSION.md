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

- the default serious evaluator path is now `strict_causal` with `doc_bias` off
- `doc_bias` is still the only hybrid feature allowed back onto the critical path, but only as a post-winner overlay
- `pointer`, `ngram`, `meta`, and adaptive extra pass are currently off the winning path
- capacity scaling is helping more than context scaling in the current proxy regime
- decimal submission-total budget accounting is now wired into serious runs
- we now have a stronger static-prior stack in `hybrid_golf`, but it still needs real remote ranking runs
- export now supports mixed-bit dense quantization and ternary BitLinear export, but final submission condensation is still missing

## Frontier Reality Check

As of **March 22, 2026**, the official accepted leaderboard in the repo README still tops out at **1.1428 BPB** with `10L Int5-MLP + BigramHash(10240)` by `thwu1`.

But the live open-PR frontier is already materially stronger:

- best validated **non-TTT** candidate tracked in the live commentary thread is **1.1233 BPB** in [PR #414](https://github.com/openai/parameter-golf/pull/414)
- best tracked **TTT** candidate is **1.1221 BPB** in [PR #398](https://github.com/openai/parameter-golf/pull/398)
- the TTT line is under legitimacy scrutiny in [Issue #140](https://github.com/openai/parameter-golf/issues/140), which explicitly notes uncertainty around pre-eval adaptation on validation tokens before scoring

Implication:

- the official README leaderboard is already stale relative to the live frontier
- our real target is not `1.1428`
- our current serious proxy numbers are still far from the live non-TTT frontier
- we should not drift into optimizing for a tiny local improvement on a weak prior

## North Star

We are not trying to win with a `+0.001 BPB` garnish on a mediocre stack.

We want a **regime change**.

Concretely:

- beat the live **non-TTT** frontier, not only the stale accepted table
- treat the challenge as **causal online compression**, not only as static-weight packing
- store global web priors in the artifact
- use only **legal causal mechanisms** to absorb document-local structure during evaluation
- first build a `#414`-class compact prior, then test whether hybrid online structure can create a real step-change on top

The strategic bet remains:

- strongest compact static prior we can afford under `16 MB`
- then a legal causal residual compressor on top of it
- not blind trick stacking

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
- config inheritance via `extends`
- challenge shard reader
- tokenizer / byte-accounting layer
- GPT model training and evaluation
- strict and exploratory evaluator policies
- ngram / pointer / doc_bias experts
- writable-state adaptation
- factorized tied embeddings
- attention / MLP bank sharing
- NormFormer-lite and depth-aware initialization
- Muon split optimizer
- SWA-style averaging
- projection-QAT and compression regularization
- recurrent shared-stack depth with pass controls and low-rank deltas
- BitLinear / ternary latent weights
- mixed-bit export (`mixed_v1` / `mixed_v2`)
- submission-total budget accounting
- roundtrip export / reload
- checkpoint-aware eval config loading

### Implemented but still simplified

- expert gate:
  heuristic and hand-tuned, not learned or calibrated
- meta-training:
  present, but currently not on the winning path
- recurrent adaptive compute:
  present, but `adaptive_extra_pass` still only drives the older top-block path, not the full recurrent branch
- branch selection workflow:
  Stage 0 winner selection and branch advancement are still manual
- submission packaging:
  code-byte accounting is real, but final single-file condensation is not yet built

### Not implemented yet from the bigger submission vision

- automatic root selection from Stage 0 into the wave-2 branch matrix
- systematic branch-ranking harness that advances only the top candidates
- full submission condensation path into a final minimal artifact script
- systematic significance / repeat-run harness
- learned or calibrated expert gating
- stronger frontier architectural branches beyond the current waves:
  partial-RoPE / LN-scaling / XSA-like additions
- optional hybrid revisits after the static-prior winner:
  pointer, writable-state methods, or meta, but only if new evidence appears

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

## Prioritized Remaining Backlog

Ordered by submission relevance, not by novelty:

1. **Stage 0 and wave-2 ranking**
   Run the already-landed branch matrix under parity-lite and full parity.

2. **Submission condensation**
   Build the final minimal artifact path rather than counting the whole repo runtime forever.

3. **Repeatability / significance**
   Add a proper rerun harness so candidate promotion is not based on one-off wins.

4. **Branch orchestration**
   Automate winner selection and finalist advancement instead of relying on manual shell flows.

5. **Hybrid re-open only after a winner exists**
   Re-test `doc_bias` on the best static prior, then decide whether any other hybrid feature deserves another pass.

6. **Further frontier architecture only if needed**
   Explore new compact-prior branches only after the current mainline stack is honestly ranked.

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
  - `doc_bias` disabled by default
  - `ngram` disabled
  - `pointer` disabled
  - `meta` disabled
  - `adaptive_extra_pass` disabled

After picking the best static prior, run exactly one overlay comparison:

- base prior
- base prior + `doc_bias`

## Phase 1: Make Every Serious Run Challenge-Aware

Status: mostly complete

What is already in place:

1. automatic export after every serious train run
2. artifact-byte reporting in train/eval summaries
3. an explicit warning or failure when export size exceeds a configured budget target
4. run labels such as:
   - `smoke`
   - `proxy`
   - `submission_candidate`

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

Remaining work in this phase:

- final submission condensation
- a stricter automated run harness that picks winners instead of relying on manual command sequences

## Phase 2: Build A Challenge-Faithful Static Prior

Status: first acceleration waves implemented, remote ranking pending

This is the biggest missing piece.

We should not keep treating the current strict baseline as if it were frontier-like. It is a clean proxy, not yet a serious static prior.

Goal:

Use the already-landed mainline stack to get a **strong compact prior** under the 16 MB export budget, then rank the new branches under parity-lite and full parity.

Already implemented in `hybrid_golf`:

1. stronger export path
   - mixed precision / mixed bit-width export
   - selective keep-float policy
   - exact treatment of small / control tensors

2. training-time robustness to export
   - projection-QAT
   - compression regularization

3. budget-aware capacity controls
   - factorized tied embeddings
   - attention / MLP bank sharing
   - Muon optimizer split
   - SWA-style averaging

4. compute-for-bytes architecture branches
   - recurrent shared-stack depth
   - pass modulation / q-gain
   - low-rank block / logit deltas

5. advanced compression branch
   - BitLinear / ternary latent weights
   - `mixed_v2` export support

What remains in this phase:

1. run Stage 0 root selection on:
   - `challenge_mainline_base`
   - `challenge_mainline_shared`
   - `challenge_mainline_shared_qat`

2. rank wave-2 branches against that root:
   - `challenge_mainline_recurrent`
   - `challenge_mainline_recurrent_delta`
   - `challenge_mainline_bitlinear_mlp`
   - `challenge_mainline_bitlinear_full`

3. advance only the top 1-2 static-prior candidates to full parity and export roundtrip

Meaning:

We do **not** need every frontier trick before doing research.
But we **do** need a strong enough static prior that hybrid gains are measured on top of something realistic.

## Phase 3: Re-test Hybrid Additions On Top Of The Strong Prior

Status: partially blocked on branch ranking

Only after the stronger prior exists should we revisit hybrid components.

Current order:

1. `doc_bias`
   - compare exactly once on the best static prior:
     - base prior
     - base prior + `doc_bias`

2. `pointer`
   - stay frozen unless the new winner leaves obvious repeat-structure on the table

3. `ngram`
   - stay frozen unless there is new evidence

4. writable-state adaptation
   - only revisit if it helps under `strict_causal`
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

1. Run Stage 0 parity-lite ranking on:
   - `challenge_mainline_base`
   - `challenge_mainline_shared`
   - `challenge_mainline_shared_qat`
2. Freeze the winning static-prior root from that Stage 0 rerun.
3. Run the wave-2 branch matrix:
   - recurrent
   - recurrent delta
   - BitLinear MLP
   - BitLinear full
4. Advance only the best 1-2 static-prior candidates to full parity plus roundtrip.
5. Re-test `doc_bias` only on the single best static prior.
6. Update the docs and runbooks with the actual winner once the ranking is real.

## Current Baseline To Beat

Internal research baseline right now:

- model family: strict baseline branch
- evaluator:
  - `strict_causal`
  - `doc_bias` disabled by default
  - `ngram` disabled
  - `pointer` disabled
  - `meta` disabled
  - `adaptive_extra_pass` disabled

Best observed historical serious proxy result so far:

- `ada_proxy12x384_seq512_w540`
- `val_bpb = 1.7747509821863527`
- validation slice: `131072` tokens
- eval regime: reduced-slice proxy (`256/512` windows), not yet parity-lite or full parity

Current perspective:

- this is our best historical serious-proxy baseline under the old reduced-slice regime
- it is still far from the live non-TTT frontier at `1.1233`
- it should no longer decide the mainline recipe without parity-lite reranking on the new stack

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
- [challenge_parity_lite.json](/c:/Users/pasqu/OpenChallenge/parameter-golf/configs/hybrid/challenge_parity_lite.json)
- [CHALLENGE_PARITY_CHECKLIST.md](/c:/Users/pasqu/OpenChallenge/parameter-golf/CHALLENGE_PARITY_CHECKLIST.md)

These define the minimum standard for treating a `hybrid_golf` result as challenge-ready rather than merely proxy-valid.

Recommended workflow:

1. Use `challenge_parity_lite.json` for candidate ranking.
   It keeps the final evaluator semantics but limits validation to a smaller slice for iteration speed.
2. Use `challenge_parity.json` only on the best 1-2 runs.
   That is the final parity gate before calling a result submission-grade.
3. Do not let lite-parity or older proxy runs decide the final recipe by themselves.
