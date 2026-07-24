# coalmine — interview prep

Know these cold. Every number regenerates from a seeded experiment in
`coalmine/experiments/`.

## The ten numbers

| # | Number | Where it comes from |
|---|---|---|
| 1 | **43.7% vs 5.9%** — repeated t-test false alarms vs calibrated CUSUM, same nominal 5% | `detection_latency` |
| 2 | **+5,566 requests** — full-loop detection latency of a 15% ε-regression (25% sampling, noisy judge); the fleet independently reproduced **+5,258** | `full_loop`, `fleet_run` |
| 3 | **49.8% vs 38.9%** — challenger win rate on equal traffic, randomized vs champion-always-first (20%-biased judge) | `position_bias` |
| 4 | **33%** — SPRT's detection rate at effects below its design alternative (mSPRT: ~99%+) | `method_comparison` |
| 5 | **~5× vs flat** — anytime methods' latency growth with a 10k healthy prefix; CUSUM stays ~1,100 verdicts | `method_comparison` (dilution) |
| 6 | **15.0% → 3.9%** — 4-arm family-wise error, uncorrected vs α/4, at +48% promotion latency | `multiarm` |
| 7 | **byte-identical** — chaos run (all 3 judge workers killed) vs baseline, ~2,700 req/s on Redis | `fleet_run` |
| 8 | **+199 requests** — drift-alarm delay after a topic-mix shift; ungated CUSUM false-blamed the model at +5,110 | `drift_gate` |
| 9 | **+1,065 requests** — auto-rollback after a mid-ramp regression; equal challenger promoted @ 10,525 | `canary_lifecycle` |
| 10 | **69.4%** — local reward-model judge vs 709 human-consensus MT-Bench pairs ($0.00); ensembles k=3/5 → effective 0.89/0.94, latency −27%/−52% | `reward_judge_study`, `ensemble_detection` |

## Questions you will get, and the answers

**Why does peeking break a t-test?** Each check controls P(false positive at
this look) = α, but acting on the *first* significant look controls
P(any look ever crosses) — which grows without bound as looks accumulate
(measured: 43.7% over 200 peeks). Sequential methods control the supremum
over time, not the per-look rate.

**Why is the promotion gate an SPRT but the alarm a CUSUM?** They answer
different questions. The SPRT tests a fixed hypothesis over the whole stream
and terminates; run from t=0 it happily accepts "no difference" before a
*late* regression starts, and anytime variants dilute — a healthy prefix
buries the shift (measured: latency grows ~5× with a 10k prefix). CUSUM is
the SPRT reflected at zero: it discards accumulated "healthy" evidence, so it
stays primed for a change starting at any time (latency flat at ~1,100
verdicts regardless of prefix). Same likelihood-ratio increments, different
machines.

**What makes the mixture SPRT valid at every sample size?** The mixture
likelihood ratio is a nonnegative martingale under H0 with expectation 1;
Ville's inequality bounds P(it ever exceeds 1/α) by α — no matter when or
how often you look. That's the whole anytime-validity trick.

**Why did you also implement a confidence sequence, and what did you find?**
The stitched boundary (Howard et al.) is prior-free and closed-form, but the
closed form is paid for in conservatism: it realized 0.27% of a 5% budget and
ran ~1.6× slower to decision than the mixture SPRT. Empirically the mixture's
prior is worth its cost.

**Walk me through the judge channel.** The tests never see truth — they see
verdicts from a judge with accuracy a: observed = w·a + (1−w)(1−a), so a true
shift δ arrives attenuated to δ·(2a−1). At a = 0.85, 5 points become 3.5.
Detection latency scales like 1/shift², so channel quality enters
*quadratically* — which is why the measured real-judge accuracy (0.694 vs
humans) tripled the planning cost, and why majority-of-k ensembles (effective
0.89 at k=3) cut latency 27–52%. Tests are therefore designed on the
observed scale; designing on the true scale silently miscalibrates them.

**How do you handle k challengers?** Per-arm anytime tests at α/k. The union
bound composes cleanly with anytime validity (each arm's supremum event has
probability ≤ α/k, so the family's ≤ α, at any k and any stopping rule).
Measured: uncorrected 4-arm FWER 15% (3× budget), corrected 3.9%, price +48%
promotion latency.

**Why non-inferiority gates on the canary ramp, not superiority?** A
superiority gate never promotes an equal-quality challenger — the realistic
case. The gate rejects "worse than a 6-point margin" (H0: p = 0.5 − δ)
upward; a truly regressed challenger fails the gate *and* trips the rollback
CUSUM that runs continuously underneath.

**How does the fleet get exactly-once without coordination?** It doesn't get
exactly-once *delivery* — Redis consumer groups give at-least-once with
reclaim. It gets exactly-once *effects*: every event's seq is derived from
request identity (10·index + offset) and every handler is a pure function of
its message (per-request RNG streams), so a redelivered message rewrites the
same (run_id, seq) row and the idempotent store drops it. Chaos runs killing
every worker converge byte-identically — verified by fingerprint.

**What does the drift gate actually prevent?** Mix-shift false blame. With
topic-heterogeneous quality, a traffic shift moves the aggregate win rate
with neither config changed; the ungated CUSUM blamed the model. The PSI
monitor caught the shift in 199 requests, reset the in-flight tests (their
iid assumption died), re-baselined — and per-topic win rates proved nothing
moved within any topic. The homogeneous control shows it's attribution, not
suppression.

**Where did your calibrations almost go wrong?** Twice, and the test suite
caught both: the PSI threshold was first calibrated with independent windows
against the true reference — the real monitor uses overlapping windows
against an *estimated* reference, which has a larger max statistic (realized
false alarms exceeded budget until the calibration simulated the monitor's
own procedure). And SQLite connections aren't safe under concurrent
`to_thread` flushes — serialized behind a shared lock.

**Why trust any of this?** Every detector's threshold is calibrated by
simulating its own procedure; every scalar implementation is cross-validated
trial-for-trial against its vectorized twin; empirical error rates are
measured against budgets, never assumed; and the judge channel is grounded in
human-labeled data. CI runs 135 deterministic seeded tests against real
Postgres and Redis on every push.

## Design decisions to defend (one-liners)

- **Synthetic verdict channel first**: FPR claims need hundreds of null runs;
  real judges make that unaffordable — so characterize the judge as a channel,
  validate statistics against the channel, measure the channel on real data.
- **ε-mixture injection**: prompt-sabotage regressions have unknown size;
  serving from a bad pool with probability ε makes ground truth exact and
  detection claims falsifiable.
- **Event log as source of truth**: verdicts, decisions, canary transitions
  all replayable; the audit trail is the database, and it doubles as the
  trace.
- **Redis deferred to Phase 5**: single-process asyncio was the honest design
  until multiple services existed; infrastructure earned its place with
  consumer groups, reclaim, and chaos semantics.
- **Reward-model judge over API judge**: position-invariant by construction,
  $0, deterministic — and calibrated against human votes rather than another
  LLM's opinion.
