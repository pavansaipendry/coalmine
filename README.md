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
.venv/bin/pytest -q                        # 51 deterministic tests (4 need Postgres)
.venv/bin/python -m coalmine.experiments.detection_latency
.venv/bin/python -m coalmine.experiments.sprt_validation
.venv/bin/python -m coalmine.experiments.epsilon_verification
.venv/bin/python -m coalmine.experiments.shadow_overhead
```

Runs on CPU in a few minutes; no GPU anywhere. The Postgres store tests skip
unless `COALMINE_PG_DSN` is set (CI provides a postgres:16 service container).

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | Decision engine: SPRT + CUSUM, judge channel model, Wald planning calculator, vectorized Monte Carlo validation | **done** |
| 2 | Traffic simulator, shadow router, ε-mixture response pools, event-sourced log (SQLite + Postgres) | **done** |
| 3 | Real judging layer: position randomization, measured position bias, sampling, judge calibration vs anchor set | next |
| 4 | mSPRT + confidence sequences; three-way method comparison; k challengers with alpha spending; stratified tests | |
| 5 | Multi-service fleet on Redis streams, load tests, chaos tests, Prometheus/Grafana/OTel | |
| 6 | Closed-loop canary lifecycle: shadow → 1% → 5% → 25% → 100% ramp with per-step gates and auto-rollback; React dashboard | |

Architecture notes and the full design rationale live in
[docs/DESIGN.md](docs/DESIGN.md).
