"""Shared Prometheus metric objects for the sentiment API.

All counters are defined here once so they are never double-registered if
modules are reloaded (e.g. during test collection). Importers increment the
counters directly; the Instrumentator in main.py exposes them at /metrics.
"""
from prometheus_client import Counter

predictions_total = Counter(
    "sentiment_predictions_total",
    "Total predictions served, partitioned by predicted label.",
    ["label"],
)

auth_rejections_total = Counter(
    "sentiment_auth_rejections_total",
    "Total requests rejected due to a missing or invalid API key.",
)

rate_limit_hits_total = Counter(
    "sentiment_rate_limit_hits_total",
    "Total requests rejected because the rate limit was exceeded.",
)
