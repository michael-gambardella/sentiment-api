"""Locust load-test suite for the Sentiment Classification API.

User classes
------------
APIUser     — realistic mixed traffic: single predict (high), batch (medium),
              health check (low). weight=10 so most simulated users are here.
ExplainUser — SHAP explain only. Kept separate because /predict/explain is
              10–50× slower; mixing it into APIUser would skew latency metrics.
              weight=1 so it represents ~9% of the user pool.

Load shape
----------
StagesShape drives an automated ramp-up → sustain → peak → sustain → ramp-down
sequence. Activated by setting env var LOAD_TEST_SHAPE=1.

Without LOAD_TEST_SHAPE, control the test from the CLI:
    locust -f load_tests/locustfile.py --host http://localhost:8000 \\
           --headless -u 20 -r 5 --run-time 60s

With shape (full benchmark, ~3.5 min):
    LOAD_TEST_SHAPE=1 locust -f load_tests/locustfile.py \\
           --host http://localhost:8000 --headless

Generate HTML report:
    locust -f load_tests/locustfile.py --host http://localhost:8000 \\
           --headless -u 20 -r 5 --run-time 60s \\
           --html load_tests/report.html --csv load_tests/results

Auth:
    Set SENTIMENT_API_KEY if the API is running with authentication enabled.
"""
from __future__ import annotations

import os
import random

from locust import HttpUser, LoadTestShape, between, task

_API_KEY = os.getenv("SENTIMENT_API_KEY", "")
_HEADERS = {"X-API-Key": _API_KEY} if _API_KEY else {}

# 20 distinct texts so the Redis cache warms up naturally after the first pass,
# mirroring real production traffic where a corpus of popular inputs recurs.
_TEXTS = [
    "This movie was absolutely fantastic!",
    "Terrible waste of time, would not recommend.",
    "An outstanding performance by the entire cast.",
    "Boring and predictable from start to finish.",
    "I laughed, I cried — a true masterpiece.",
    "The plot made no sense whatsoever.",
    "Visually stunning with a gripping story.",
    "Complete disappointment compared to the original.",
    "One of the best films I have seen in years.",
    "Slow pacing killed what could have been a good film.",
    "Heartwarming and beautifully acted throughout.",
    "The dialogue was wooden and unconvincing.",
    "A must-see for fans of the genre.",
    "I fell asleep halfway through the second act.",
    "Exceptional direction and breathtaking cinematography.",
    "The ending was a complete and utter letdown.",
    "Thoroughly enjoyed every single moment of it.",
    "Not worth the ticket price at all.",
    "A delightful surprise — highly recommended to everyone.",
    "Mediocre at best, entirely forgettable at worst.",
]

# Batch sizes weighted toward smaller payloads (common in practice).
_BATCH_SIZES = [2, 2, 4, 4, 8, 16]


class APIUser(HttpUser):
    """Simulates a typical API consumer.

    Task weights reflect realistic usage: single predictions dominate,
    batch calls are occasional, health checks are background monitoring.
    """

    weight = 10
    wait_time = between(0.25, 1.0)

    @task(20)
    def predict_single(self) -> None:
        text = random.choice(_TEXTS)
        with self.client.post(
            "/predict",
            json={"text": text},
            headers=_HEADERS,
            catch_response=True,
            name="/predict",
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:120]}")

    @task(5)
    def predict_batch(self) -> None:
        size = random.choice(_BATCH_SIZES)
        texts = random.choices(_TEXTS, k=size)
        with self.client.post(
            "/predict/batch",
            json={"texts": texts},
            headers=_HEADERS,
            catch_response=True,
            name="/predict/batch",
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:120]}")
            else:
                data = resp.json()
                if data.get("count") != size:
                    resp.failure(f"Expected {size} results, got {data.get('count')}")

    @task(2)
    def health_check(self) -> None:
        self.client.get("/health", name="/health")


class ExplainUser(HttpUser):
    """Isolated user class for /predict/explain load.

    Long wait_time and high request timeout because SHAP requires many model
    calls (~1–10 s per request). Kept separate so its latency does not inflate
    the APIUser statistics.
    """

    weight = 1
    wait_time = between(5.0, 15.0)

    @task
    def predict_explain(self) -> None:
        text = random.choice(_TEXTS[:10])  # shorter texts finish faster
        with self.client.post(
            "/predict/explain",
            json={"text": text},
            headers=_HEADERS,
            catch_response=True,
            name="/predict/explain",
            timeout=30,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:120]}")


# ---------------------------------------------------------------------------
# Optional automated load shape — activated by LOAD_TEST_SHAPE=1
# ---------------------------------------------------------------------------
# Stage breakdown (elapsed → target users, spawn rate):
#   0 –  30 s :  10 users,  1/s  — warm-up (fills Redis cache)
#  30 –  90 s :  10 users, 10/s  — baseline measurement window
#  90 – 120 s :  50 users, 13/s  — ramp to peak
# 120 – 180 s :  50 users, 50/s  — peak measurement window
# 180 – 210 s :   0 users, 17/s  — ramp down / graceful finish
# ---------------------------------------------------------------------------

if os.getenv("LOAD_TEST_SHAPE", "").lower() in ("1", "true", "yes"):

    class StagesShape(LoadTestShape):
        _stages = [
            (30,  10,  1),
            (90,  10, 10),
            (120, 50, 13),
            (180, 50, 50),
            (210,  0, 17),
        ]

        def tick(self) -> tuple[int, float] | None:
            t = self.get_run_time()
            for end, users, rate in self._stages:
                if t <= end:
                    return users, rate
            return None  # ends the test
