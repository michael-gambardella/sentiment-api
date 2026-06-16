"""OpenTelemetry tracing setup.

Tracing is opt-in: set OTEL_EXPORTER_OTLP_ENDPOINT to enable it.
When the env var is absent (CI, local dev without a collector) this module
is a no-op — no imports from the SDK fail and no side-effects occur.
"""
import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def configure_tracing(app) -> None:
    """Wire up the global TracerProvider and instrument the FastAPI app.

    Does nothing when OTEL_EXPORTER_OTLP_ENDPOINT is unset so that unit
    tests and local development without a collector work without changes.
    """
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return

    resource = Resource.create(
        {SERVICE_NAME: os.getenv("OTEL_SERVICE_NAME", "sentiment-api")}
    )
    provider = TracerProvider(resource=resource)
    # OTLPSpanExporter reads OTEL_EXPORTER_OTLP_ENDPOINT from the environment;
    # insecure=True disables TLS for intra-Docker HTTP communication.
    exporter = OTLPSpanExporter(insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    FastAPIInstrumentor.instrument_app(
        app,
        excluded_urls="/health,/metrics",
    )
