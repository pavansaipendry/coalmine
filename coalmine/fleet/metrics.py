"""Prometheus metrics for the fleet — a monitoring system should be monitored.

Counters and histograms per pipeline stage, exposed via prometheus_client.
Metric objects are module-level singletons (Prometheus registries reject
duplicate registration), labeled by service."""

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    start_http_server,
)

REGISTRY = CollectorRegistry()

MESSAGES_PROCESSED = Counter(
    "coalmine_messages_processed_total",
    "Messages processed per service",
    ["service"],
    registry=REGISTRY,
)
EVENTS_WRITTEN = Counter(
    "coalmine_events_written_total",
    "Events appended to the store per service",
    ["service"],
    registry=REGISTRY,
)
WORKER_RESTARTS = Counter(
    "coalmine_worker_restarts_total",
    "Judge worker kill/restart cycles (chaos)",
    registry=REGISTRY,
)
ALARMS = Counter(
    "coalmine_alarms_total",
    "Alarms raised, by kind",
    ["kind"],
    registry=REGISTRY,
)
STAGE_LATENCY = Histogram(
    "coalmine_stage_latency_seconds",
    "Per-message processing latency by service",
    ["service"],
    registry=REGISTRY,
    buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 1.0),
)
STREAM_PENDING = Gauge(
    "coalmine_stream_pending",
    "Pending (unacked) messages per stream/group",
    ["stream", "group"],
    registry=REGISTRY,
)


def scrape() -> bytes:
    """Current metrics in Prometheus exposition format."""
    return generate_latest(REGISTRY)


def serve(port: int = 9108) -> None:
    """Expose /metrics for Prometheus to scrape (see deploy/prometheus.yml)."""
    start_http_server(port, registry=REGISTRY)
