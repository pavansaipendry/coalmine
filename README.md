# coalmine

**An automated canary control plane for LLM configs** — shadow-tests challenger
configurations against live traffic and promotes or rolls back via sequential
hypothesis testing, with every decision statistically calibrated and auditable.

The problem: teams ship a new prompt, model version, or sampling config, then
watch a dashboard and eyeball whether quality moved. Checking results
continuously and acting when they "look significant" is statistically broken —
repeated testing inflates false alarms far past the nominal rate. coalmine
replaces the eyeball with sequential tests that hold their error budgets under
continuous monitoring, and closes the loop: shadow → gated canary ramp →
promote or auto-rollback.

## Results so far (Phase 1: the decision engine)

All numbers from seeded Monte Carlo runs — reproduce with the commands below.

**Peeking is broken; the sequential test is not.** On identical null streams
(no regression, 20,000 pairwise verdicts), a repeated t-test peeking every 100
verdicts false-alarms **43.7%** of the time despite claiming 5% per check. A
CUSUM with its threshold calibrated to a 5% budget over the same horizon
realizes **5.9%**.

![false alarm comparison](experiments/results/false_alarm_comparison.png)

**Detection latency vs regression size, at that controlled false-alarm rate.**
Regressions are injected at a known changepoint with exact ground-truth size;
verdicts arrive through a judge modeled as a noisy channel (accuracy 0.85, tie
rate 0.10), so a true 3-point win-rate drop reaches the detector as a 2.1-point
observed drop. Median detection: a 5-point true regression in ~2,000 verdicts,
a 3-point one in ~4,800 (caught 84% of runs within a 15,000-verdict budget).

![detection latency](experiments/results/detection_latency.png)

**The implementation matches Wald's theory.** Empirical promote-probability and
expected-sample-size curves sit on the closed-form predictions across the whole
operating range, with the small upward gap in E[N] that boundary overshoot
predicts. This is the evidence the engine is correct, not just plausible.

![sprt validation](experiments/results/sprt_validation.png)

## Results so far (Phase 2: traffic, shadowing, the event log)

**Shadowing costs the user nothing — measured, not asserted.** Three paced runs
(1,200 requests at 60 rps) share identical seeded arrivals and champion latency
draws; only the shadow rate varies. With 100% of traffic shadowed to a
challenger 50% *slower* than the champion, p99 user-facing latency moves from
215.8ms to 216.0ms — 0.2ms, within timer noise. Shadow calls are fired as
tasks that the user path never awaits.

![shadow overhead](experiments/results/shadow_overhead.png)

**Injected ground truth is recoverable from the event log.** A 20,000-request
run injects a 10% ε-mixture regression into the challenger at request 10,000.
Replaying the append-only log recovers ε = 0.000 before the changepoint and
0.099 after, against 0.10 injected — the property that makes every future
detection claim scorable against exact truth. Same-seed runs produce identical
order-free content fingerprints (response content is a pure function of
(seed, request, config); scheduler interleaving is canonicalized away).

![epsilon verification](experiments/results/epsilon_verification.png)

Storage is event-sourced behind one interface with two backends — SQLite
(zero-setup dev) and Postgres (the fleet backend, exercised in CI against a
real service container). Appends are idempotent on (run_id, seq), so
at-least-once delivery yields exactly-once storage: the chaos-testing phase is
safe by construction.

## Results so far (Phase 3: the judging layer, and the first full loop)

**The full loop catches an injected regression end to end.** First composition
of all three phases: 30,000 simulated requests flow through the shadow router;
a noisy position-randomized judge (accuracy 85%, ties 10%, position bias 15%)
turns the 25% shadow-sampled pairs into verdicts; the calibrated CUSUM watches
the decisive-verdict stream. A 15% ε-mixture regression injected at request
10,000 drags the observed win rate from 49.8% to 44.0%; the alarm fires 5,566
requests later, while the paired null run stays quiet for all 30,000 requests.
Traffic, verdicts, and detection all live in one replayable event stream.

![full loop](experiments/results/full_loop.png)

**Position bias: measured, and neutralized by randomization.** The same
equal-quality responses judged by the same judge with a 20% first-position
preference: randomized presentation preserves parity (49.8%) and exposes the
bias in position space (+9.1 points toward first-shown); always showing the
champion first silently corrupts the challenger's measured win rate to 38.9%.
Every real comparison in this system randomizes — and records the order, so
the bias stays measurable.

![position bias](experiments/results/position_bias.png)

**The system monitors its own sensor.** Every detection guarantee is
conditioned on judge accuracy, so the judge is re-calibrated each round
against a frozen anchor set with known labels. When the judge's true accuracy
silently degrades from 0.85 to 0.72, the Wilson interval crosses the design
tolerance in the very round the degradation begins — zero false alarms in the
ten healthy rounds before it. (One post-degradation round briefly recovers
above threshold on sampling noise; alarm persistence rules are future work,
and honestly noted.)

![judge calibration](experiments/results/judge_calibration.png)

The production judge (`coalmine.judging.llm.LLMJudge`, Claude Haiku via
structured outputs, temperature 0) implements the same protocol as the
simulation oracle and sees only query and response texts — never config
identity or ground-truth labels. Per the cost discipline in the design, the
big experiments never call it: it exists for the live demo path and the
anchor-set calibration study that estimates the channel parameters.

## Results so far (Phase 4: three sequential methods, k arms, stratification)

**Three-way method comparison — each method's real trade-off, measured.**
Wald's SPRT, the one-sided mixture SPRT (the always-valid method Optimizely
industrialized), and the Howard-et-al stitched confidence sequence, all at a
nominal 5%. SPRT is fastest at its design point but its detection rate
collapses to 33% at effects smaller than the point alternative it was tuned
for; the mixture SPRT detects reliably at every effect size with no
alternative to tune, at ~1.6× lower latency than the stitched CS; the
stitched boundary realizes only 0.27% of its 5% budget — the closed form is
paid for in conservatism. All three hold the error budget.

![method comparison](experiments/results/method_comparison.png)

**Why the alarm is a CUSUM and not an anytime test.** The always-valid
methods test a fixed hypothesis over the whole stream, so a healthy prefix
buries a late regression: their detection latency grows ~5× as the prefix
grows to 10,000 verdicts, while CUSUM — which restarts its evidence at zero —
stays flat at ~1,100 verdicts regardless. Promotion gates and regression
alarms are different primitives; this chart is the reason.

![changepoint dilution](experiments/results/changepoint_dilution.png)

**Four challengers without alpha spending is peeking again.** Watching 4 null
challengers each at α = 5% promotes some not-better arm 15.0% of the time —
3× the budget. Splitting the budget (α/4 per arm, valid at any k by union
bound over anytime tests) holds the family-wise error at 3.9%, and the
measured price is +48% promotion latency on a truly better arm.

![multiarm](experiments/results/multiarm.png)

**Stratified monitors catch what the aggregate smears out.** A regression
confined to one topic (25% of traffic) is diluted 4:1 in the aggregate win
rate but violent inside its stratum. Per-topic CUSUMs at α/4 — reusing the
topic labels already in the event log — catch it 1.9× sooner than the
aggregate monitor (+5,819 vs +10,977 requests), with zero false alarms on
healthy topics or the paired null run. Run end to end through the real
pipeline: traffic → judge → per-stratum detection.

![stratified](experiments/results/stratified.png)

## Results so far (Phase 5: the fleet, chaos, drift gating, the ensemble)

**The fleet on real Redis, with every worker killed — and nothing changes.**
The multi-service pipeline (traffic → router → 3 judge workers in one
consumer group → decision engine, with the drift monitor alongside) sustains
~2,700 req/s over Redis Streams. In the chaos run all three judge workers are
killed mid-flight; unacked messages are reclaimed by the survivors, and
because every effect is idempotent (event seqs derived from request identity)
and deterministic (per-request RNG streams), the run converges to
byte-identical verdicts and the identical alarm at request 13,258 — verified
by fingerprint. The fleet's detection latency independently reproduces
Phase 3's in-process result on the same statistical setup.

![fleet run](experiments/results/fleet_run.png)

**The drift gate: "traffic changed, not the model."** The challenger has a
stable weakness on one topic; at request 15,000 the traffic mix shifts toward
that topic and the aggregate win rate drops — with neither config changed. An
ungated CUSUM blames the model (false quality alarm at 20,110). The drift
monitor sees the mix shift in 199 requests, fires first, resets the in-flight
test and re-baselines; per-topic win rates confirm nothing moved within any
topic (coding: 45.7% → 44.9%). A homogeneous control shows the gate isn't
suppression: its drift alarm fires, its quality alarm never does.

![drift gate](experiments/results/drift_gate.png)

**The ensemble absorbs a failing judge — and notices it without labels.**
When one of three judges silently degrades (0.85 → 0.55), majority voting
holds ensemble accuracy at 86.6% while the bad member falls to coin-flip
territory — and the members' disagreement rate jumps past its calibrated
threshold 249 comparisons after the degradation begins, a sensor alarm that
needs no ground truth and fires long before the next scheduled anchor-set
calibration round.

![judge ensemble](experiments/results/judge_ensemble.png)

The fleet is observable end to end: Prometheus counters and histograms per
stage (`coalmine.fleet.metrics`), and `deploy/docker-compose.yml` brings up
Redis, Postgres, Prometheus, and a provisioned Grafana dashboard for local
runs. The event log doubles as the trace — every request's path through the
pipeline is reconstructible from its events.

## Design principles

- **Verdict-channel abstraction.** The decision engine consumes win/loss/tie
  streams and does not care where they come from. Monte Carlo experiments use a
  synthetic generator; production uses an LLM judge. The judge is modeled as a
  noisy channel (accuracy, tie rate) whose parameters are measured, and tests
  are designed on the observed scale — judge error attenuates every effect by
  2·accuracy − 1, and ignoring that miscalibrates the tests.
- **Exact ground truth via ε-mixture injection.** Simulated regressions serve
  from a deliberately-bad response pool with probability ε, so the true effect
  size is exact by construction and "detected the 3% regression" is a
  measurable claim.
- **Two primitives, two jobs.** The SPRT is the promotion gate (fixed
  hypothesis, decide once); Page's CUSUM — the SPRT reflected at zero — is the
  regression alarm (a change starting at an unknown time). Same likelihood
  ratios, different machines.
- **Two-tier CI.** Every test in `tests/` is seeded and deterministic — a
  failure is a regression, never an unlucky draw. Fresh-seed statistical soaks
  run on manual dispatch.

## Repro

```bash
uv venv .venv && uv pip install -e ".[dev]" -p .venv/bin/python
.venv/bin/pytest -q                        # 118 deterministic tests (Postgres + Redis in CI)
.venv/bin/python -m coalmine.experiments.detection_latency
.venv/bin/python -m coalmine.experiments.sprt_validation
.venv/bin/python -m coalmine.experiments.epsilon_verification
.venv/bin/python -m coalmine.experiments.shadow_overhead
.venv/bin/python -m coalmine.experiments.full_loop
.venv/bin/python -m coalmine.experiments.position_bias
.venv/bin/python -m coalmine.experiments.judge_calibration
.venv/bin/python -m coalmine.experiments.method_comparison
.venv/bin/python -m coalmine.experiments.multiarm
.venv/bin/python -m coalmine.experiments.stratified
.venv/bin/python -m coalmine.experiments.fleet_run       # uses local Redis if present
.venv/bin/python -m coalmine.experiments.drift_gate
.venv/bin/python -m coalmine.experiments.judge_ensemble
```

Runs on CPU in a few minutes; no GPU anywhere. The Postgres store tests skip
unless `COALMINE_PG_DSN` is set (CI provides a postgres:16 service container).

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | Decision engine: SPRT + CUSUM, judge channel model, Wald planning calculator, vectorized Monte Carlo validation | **done** |
| 2 | Traffic simulator, shadow router, ε-mixture response pools, event-sourced log (SQLite + Postgres) | **done** |
| 3 | Judging layer: position randomization + measured bias, sampling, anchor-set calibration, LLM judge, full-loop detection | **done** |
| 4 | mSPRT + stitched confidence sequence; three-way comparison; k challengers with alpha spending; stratified tests | **done** |
| 5 | Redis Streams fleet; chaos-verified exactly-once effects; drift monitor gating the decision engine; judge ensemble w/ disagreement-as-drift; Prometheus + Grafana | **done** |
| 6 | Closed-loop canary lifecycle: shadow → 1% → 5% → 25% → 100% ramp with per-step gates and auto-rollback; React dashboard | next |

Architecture notes and the full design rationale live in
[docs/DESIGN.md](docs/DESIGN.md).
