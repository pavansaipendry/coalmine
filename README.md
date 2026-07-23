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
.venv/bin/pytest -q                                      # 23 deterministic tests
.venv/bin/python -m coalmine.experiments.detection_latency
.venv/bin/python -m coalmine.experiments.sprt_validation
```

Runs on CPU in a few minutes; no GPU anywhere in this phase.

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | Decision engine: SPRT + CUSUM, judge channel model, Wald planning calculator, vectorized Monte Carlo validation | **done** |
| 2 | Traffic simulator, shadow router, response pools, event-sourced Postgres log | next |
| 3 | Real judging layer: position randomization, measured position bias, sampling, judge calibration vs anchor set | |
| 4 | mSPRT + confidence sequences; three-way method comparison; k challengers with alpha spending; stratified tests | |
| 5 | Multi-service fleet on Redis streams, load tests, chaos tests, Prometheus/Grafana/OTel | |
| 6 | Closed-loop canary lifecycle: shadow → 1% → 5% → 25% → 100% ramp with per-step gates and auto-rollback; React dashboard | |

Architecture notes and the full design rationale live in
[docs/DESIGN.md](docs/DESIGN.md).
