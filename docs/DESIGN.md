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

Implemented in Phase 4:

- **Mixture SPRT** (one-sided, Beta prior truncated above p0, closed form via
  the regularized incomplete beta): always-valid by Ville's inequality, no
  point alternative to tune. Empirically: detects reliably at every effect
  size where plain SPRT's detection rate collapses off its design point.
- **Stitched confidence sequence** (Howard et al. closed-form boundary):
  prior-free but conservative — realized 0.27% of a 5% budget, ~1.6× slower
  than the mixture. Every method's realized error is measured by Monte Carlo,
  never trusted from formulas.
- **Changepoint dilution**: anytime methods test a fixed hypothesis, so a
  healthy prefix buries a late regression (latency grows ~5× with a 10k
  prefix) while CUSUM stays flat. Promotion gate and regression alarm are
  different primitives — measured, not asserted.
- **k challengers**: per-arm always-valid tests at α/k give family-wise error
  ≤ α at any k by union bound. Measured: uncorrected 4-arm FWER 15% vs 3.9%
  corrected, at +48% promotion latency.
- **Stratified monitors**: per-topic CUSUMs at α/k reuse the log's topic
  labels (embedding clusters, once real datasets arrive) and catch a
  one-topic regression ~2× sooner than the aggregate.

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

Implemented in Phase 5: multi-service topology on Redis Streams with consumer
groups (traffic, router, judge workers, drift monitor, decision engine) —
`coalmine/fleet/`. Exactly-once effects without coordination: event seqs are
derived from request identity (seq bands) and every handler is a pure
function of its message, so at-least-once delivery + idempotent appends
converge; chaos runs (every judge worker killed mid-flight) verify
byte-identical final state by fingerprint, over both the real Redis bus and
its in-memory twin. The drift monitor (windowed PSI over topic labels,
threshold calibrated by simulating the monitor's own sliding-window procedure
— independent-window calibration understates the max statistic and the test
suite caught exactly that) gates the decision engine: a drift alarm resets
in-flight sequential tests and triggers re-baselining. Prometheus metrics per
stage; `deploy/docker-compose.yml` provisions Redis/Postgres/Prometheus/
Grafana with a committed dashboard. The event log doubles as the trace.

Remaining for Phase 6: the canary lifecycle state machine over this fleet,
HTTP surface + React dashboard (SSE), k6 load tests against that surface, and
Postgres partitioning for long-horizon soaks.

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
