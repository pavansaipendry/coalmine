from coalmine.fleet import metrics


def _value(name: str, labels: dict) -> float:
    return metrics.REGISTRY.get_sample_value(name, labels) or 0.0


def test_counters_and_scrape():
    # The registry is process-global and other tests (the fleet) increment it
    # too — assert deltas, never absolutes.
    before = _value("coalmine_alarms_total", {"kind": "quality_alarm"})
    metrics.MESSAGES_PROCESSED.labels("router").inc()
    metrics.ALARMS.labels("quality_alarm").inc(2)
    metrics.STAGE_LATENCY.labels("judge").observe(0.003)
    after = _value("coalmine_alarms_total", {"kind": "quality_alarm"})
    assert after - before == 2.0
    text = metrics.scrape().decode()
    assert 'coalmine_messages_processed_total{service="router"}' in text
    assert "coalmine_stage_latency_seconds_bucket" in text
