# coalmine — design

One sentence: a system that watches two (or k) model configurations serve live
traffic and decides, with statistical guarantees, whether a challenger is safe
to promote — alarming when production quality or traffic drifts away from what
offline evals covered, and executing the promotion/rollback itself.

## Why the decision layer is the product

Observability dashboards for LLMs exist (Langfuse, Arize). None of them make
*calibrated decisions*. The differentiator is a decision engine whose false
alarm and miss rates are controlled quantities under continuous monitoring —
plus the closed promotion loop built on top of it. When time is short,
dashboard polish is cut; statistics experiments are never cut.

## Components

1. **Traffic simulator** — replays real datasets as request streams with
   realistic arrival patterns. Knobs: inject a quality regression at time T
   (ε-mixture, below), shift topic distribution, spike adversarial inputs.
   This is the ground-truth generator.
2. **Shadow router** — every request served by the champion; a sampled
   fraction also sent to challengers, fire-and-forget async, zero user-facing
   latency (proven by load test, not asserted).
3. **Judging layer** — paired comparison, not absolute scores. Position
   randomized (and the position bias measured); temperature 0, strict JSON
   rubric; judged fraction is a knob in the power calculation. Judge ensemble
   later, with disagreement rate as its own drift signal.
4. **Decision engine** — sequential tests that tolerate continuous peeking
   (this repo's Phase 1; see below).
5. **Drift monitors + dashboard** — input drift via embedding-distribution
   tests over sliding windows (with their own false-alarm rate measured under
   the null — repeated MMD tests have the same peeking pathology the decision
   engine exists to fix); output drift via win rates, lengths, refusal rates.
   An input-drift alarm *gates* the decision engine: in-flight sequential
   tests are invalidated and restarted, because their iid assumption died.

## The statistics

### Peeking

Checking a classical test repeatedly inflates false alarms (measured here:
43.7% for a nominal-5% t-test peeked every 100 verdicts over a 20k-verdict
horizon). Everything downstream depends on tests that are valid under
continuous monitoring.

### Two primitives

- **SPRT (promotion gate).** H0: observed win rate p0 vs H1: p1, decided once.
  Wald thresholds log((1−β)/α), log(β/(1−α)); guarantees hold up to boundary
  overshoot (empirically: realized error slightly conservative, E[N] slightly
  above theory — both confirmed by the validation experiment).
- **Page's CUSUM (regression alarm).** The same LLR increments with the
  statistic reflected at 0 — stays primed to catch a change beginning at any
  time. Threshold h calibrated by simulation: h = (1−rate)-quantile of the
  null running-max distribution over the monitoring horizon.

Planned additions: mSPRT (Optimizely's industrialized mixture SPRT) and one
time-uniform confidence-sequence method (Howard et al.), compared empirically
against SPRT on detection latency and realized false alarms; k challengers
need alpha spending across arms on top of sequential validity.

### The judge is a sensor, and sensors lie

Verdicts are judge opinions, not truth. Model the judge as a noisy channel:
accuracy a, tie rate t. Ties are dropped (tests run on decisive verdicts);
a true win rate w is observed as w·a + (1−w)(1−a), so every true shift is
attenuated by 2a−1. Tests are therefore designed on the *observed* scale.
Consequences, all confirmed in Phase 1 experiments:

- At a = 0.85, a 3-point true regression is a 2.1-point observed one — this,
  not the test, is why small regressions are expensive to catch.
- The planning calculator (`coalmine.stats.wald.plan`) converts a design
  (true effect, α, β, judge accuracy, sampling rate) into expected verdicts
  and live requests — the principled answer to "what sampling rate?".
- Later: the judge is itself monitored against a frozen anchor set (MT-Bench
  human judgments); judge-accuracy drift triggers a sensor alarm. A judge
  ensemble's disagreement rate is an earlier warning still.

### ε-mixture injection

Prompt-sabotage "regressions" have unknown effect size, making detection
claims unfalsifiable. Instead: pre-generate good and deliberately-bad response
pools per query; a regressed challenger serves from the bad pool with
probability ε. Ground truth is exactly ε, runs are deterministic replays, and
generation cost is paid once.

### Monte Carlo discipline

FPR claims need hundreds of null runs (50 runs → ±6% CI on a 5% rate: useless).
Hence the synthetic verdict channel: the vectorized harness
(`coalmine/sim/runner.py`) runs thousands of trials as numpy sweeps, and is
cross-validated trial-for-trial against the scalar implementations in tests.

### Determinism contract (Phase 2, implemented)

Response content is a pure function of (seed, request_index, config): every
random draw comes from an RNG keyed by those three, never a shared generator.
Same-seed runs are therefore content-identical even though concurrent shadow
tasks interleave nondeterministically — replay comparisons canonicalize by
sorting payloads (order-free fingerprints). Content is guaranteed; global
event ordering deliberately is not.

## Systems layer (Phases 2, 5, 6)

Multi-service topology on Redis streams with consumer groups (traffic gen,
shadow router, judge workers, decision engine, drift monitor), horizontally
scalable judge workers, docker-compose fleet. Postgres event-sourced log
(partitioned) with deterministic replay — new detector versions are backtested
against recorded history. k6 load tests publish the p50/p99 shadow-overhead
numbers. Chaos: kill workers mid-run, restart Redis, verify identical final
decisions (idempotent, at-least-once processing). Prometheus + Grafana + OTel
throughout; React dashboard reads the same metrics via SSE.

## The capstone loop (Phase 6)

Canary lifecycle state machine: challenger at shadow (0% user traffic) →
sequential gate passes → ramp 1% → 5% → 25% → 100%, a fresh sequential test
gating every step → auto-rollback on any post-promotion regression, full audit
trail event-sourced. Demo: inject a regression mid-ramp, watch the system
catch it, roll back, and print the audit log of every decision it made.

## Two-tier testing

- `tests/`: seeded, deterministic — failures are regressions, never unlucky
  draws. Includes exact scalar-vs-vectorized cross-validation.
- Statistical soak (manual dispatch in CI): fresh seeds, tolerance bands, both
  experiments end-to-end with artifacts uploaded.

## Constraints

- All Phase 1–4 compute is CPU + API calls; runs on a laptop. Any future
  GPU need (self-hosted pool generation) goes to a rented pod, never local.
- Judge cost discipline: the big experiments never call a real judge; the
  real judge is used for the live demo path and to *estimate* channel
  parameters from a small calibration study.

## Reading list

Wald (1945) sequential analysis; Page (1954) CUSUM; Howard et al.,
time-uniform confidence sequences; Evan Miller, "How Not To Run an A/B Test";
Netflix tech blog on sequential testing; MT-Bench paper for judge agreement
rates.
