"""OpenTelemetry instrumentation for Raspberry Pi Client."""

import logging
import os
from typing import Optional

from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION, DEPLOYMENT_ENVIRONMENT


def get_resource(device_id: str) -> Resource:
    """Create resource with service and device information."""
    return Resource.create({
        SERVICE_NAME: "raspberry-pi-client",
        SERVICE_VERSION: "1.0.0",
        DEPLOYMENT_ENVIRONMENT: os.getenv("ENV", "production"),
        "service.namespace": "kin-voice-ai",
        "device.id": device_id,
        "device.type": "raspberry-pi",
    })


def setup_tracing(device_id: str, endpoint: str = "http://localhost:4318") -> Optional[TracerProvider]:
    """Setup OpenTelemetry tracing.
    
    Args:
        device_id: Device ID
        endpoint: OTLP endpoint (local collector)
        
    Returns:
        TracerProvider instance or None if disabled
    """
    if not os.getenv("OTEL_ENABLED", "true").lower() == "true":
        return None
    
    resource = get_resource(device_id)
    provider = TracerProvider(resource=resource)
    
    # OTLP exporter for traces (to local collector)
    otlp_exporter = OTLPSpanExporter(
        endpoint=f"{endpoint}/v1/traces",
        timeout=30,
    )
    
    # Batch processor with aggressive batching for Pi
    processor = BatchSpanProcessor(
        otlp_exporter,
        max_queue_size=1024,
        max_export_batch_size=256,
        schedule_delay_millis=10000,  # 10 seconds
    )
    
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    
    return provider


def setup_metrics(device_id: str, endpoint: str = "http://localhost:4318") -> Optional[MeterProvider]:
    """Setup OpenTelemetry metrics.
    
    Args:
        device_id: Device ID
        endpoint: OTLP endpoint (local collector)
        
    Returns:
        MeterProvider instance or None if disabled
    """
    if not os.getenv("OTEL_ENABLED", "true").lower() == "true":
        return None
    
    resource = get_resource(device_id)
    
    # OTLP exporter for metrics (to local collector)
    otlp_exporter = OTLPMetricExporter(
        endpoint=f"{endpoint}/v1/metrics",
        timeout=30,
    )
    
    # Metric reader with longer interval for Pi
    reader = PeriodicExportingMetricReader(
        otlp_exporter,
        export_interval_millis=60000,  # Export every 60 seconds
    )
    
    provider = MeterProvider(
        resource=resource,
        metric_readers=[reader],
    )
    
    metrics.set_meter_provider(provider)
    
    return provider


def setup_logging(device_id: str, endpoint: str = "http://localhost:4318") -> Optional[LoggerProvider]:
    """Setup OpenTelemetry logging.
    
    Args:
        device_id: Device ID
        endpoint: OTLP endpoint (local collector)
        
    Returns:
        LoggerProvider instance or None if disabled
    """
    if not os.getenv("OTEL_ENABLED", "true").lower() == "true":
        return None
    
    resource = get_resource(device_id)
    provider = LoggerProvider(resource=resource)
    
    # OTLP exporter for logs (to local collector)
    otlp_exporter = OTLPLogExporter(
        endpoint=f"{endpoint}/v1/logs",
        timeout=30,
    )
    
    # Batch processor for logs
    processor = BatchLogRecordProcessor(
        otlp_exporter,
        max_queue_size=1024,
        max_export_batch_size=256,
        schedule_delay_millis=10000,  # 10 seconds
    )
    
    provider.add_log_record_processor(processor)
    
    # Add handler to root logger
    handler = LoggingHandler(level=logging.INFO, logger_provider=provider)
    logging.getLogger().addHandler(handler)
    
    return provider


def setup_telemetry(device_id: str, endpoint: str = "http://localhost:4318"):
    """Setup all OpenTelemetry components.
    
    Args:
        device_id: Device ID
        endpoint: OTLP endpoint (local collector on Pi)
    """
    if not os.getenv("OTEL_ENABLED", "true").lower() == "true":
        logging.info("OpenTelemetry is disabled")
        return None, None, None
    
    logging.info(
        f"Initializing OpenTelemetry for raspberry-pi-client, "
        f"device_id: {device_id}, "
        f"endpoint: {endpoint}"
    )
    
    # Setup providers
    tracer_provider = setup_tracing(device_id, endpoint)
    meter_provider = setup_metrics(device_id, endpoint)
    logger_provider = setup_logging(device_id, endpoint)
    
    logging.info("OpenTelemetry initialized successfully")
    
    return tracer_provider, meter_provider, logger_provider


def get_tracer(name: str):
    """Get a tracer instance.
    
    Args:
        name: Name of the tracer (typically __name__)
        
    Returns:
        Tracer instance
    """
    return trace.get_tracer(name)


def get_meter(name: str):
    """Get a meter instance.
    
    Args:
        name: Name of the meter (typically __name__)
        
    Returns:
        Meter instance
    """
    return metrics.get_meter(name)


def get_logger(name: str, device_id: str = None):
    """Get a logger instance with structured logging support.
    
    Args:
        name: Logger name (typically __name__)
        device_id: Device ID to include in all logs
        
    Returns:
        Logger instance configured with OTEL handler
    """
    logger = logging.getLogger(name)
    
    # Create custom adapter that adds device_id to all log records
    if device_id:
        class DeviceAdapter(logging.LoggerAdapter):
            def process(self, msg, kwargs):
                # Add OTEL attributes to every log record
                extra = kwargs.get('extra', {})
                extra['device_id'] = device_id
                extra['service.name'] = "raspberry-pi-client"
                extra['service.namespace'] = "kin-voice-ai"
                extra['telemetry.sdk.language'] = "python"
                kwargs['extra'] = extra
                return msg, kwargs

        return DeviceAdapter(logger, {'device_id': device_id})
    
    return logger


def add_span_attributes(**attributes):
    """Add attributes to the current span.
    
    Args:
        **attributes: Key-value pairs to add as span attributes
    """
    span = trace.get_current_span()
    if span.is_recording():
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(key, str(value))


def add_span_event(name: str, **attributes):
    """Add an event to the current span.
    
    Args:
        name: Event name
        **attributes: Event attributes
    """
    span = trace.get_current_span()
    if span.is_recording():
        span.add_event(name, attributes)


def record_exception(exception: Exception):
    """Record an exception in the current span.
    
    Args:
        exception: Exception to record
    """
    span = trace.get_current_span()
    if span.is_recording():
        span.record_exception(exception)
        span.set_status(trace.Status(trace.StatusCode.ERROR, str(exception)))


def create_span(name: str, **attributes):
    """Create a new span for manual instrumentation.
    
    Args:
        name: Span name
        **attributes: Span attributes
        
    Returns:
        Span context manager
        
    Example:
        with create_span("audio_processing", user_id=user_id):
            # Your code here
            pass
    """
    tracer = get_tracer(__name__)
    span = tracer.start_span(name)
    
    # Add attributes
    for key, value in attributes.items():
        if value is not None:
            span.set_attribute(key, str(value))
    
    return trace.use_span(span, end_on_exit=True)


def create_conversation_trace(name: str, **attributes):
    """Create a new root trace for a conversation.
    
    This creates a completely new trace (not a child of any existing trace)
    for conversation-level tracking across services.
    
    Args:
        name: Span name (e.g., "conversation")
        **attributes: Span attributes
        
    Returns:
        Span context manager
        
    Example:
        with create_conversation_trace("conversation", conversation_id=conv_id):
            # Conversation code here
            pass
    """
    tracer = get_tracer(__name__)
    # Start a new span without parent context (new root trace)
    span = tracer.start_span(name)
    
    # Add attributes
    for key, value in attributes.items():
        if value is not None:
            span.set_attribute(key, str(value))
    
    return trace.use_span(span, end_on_exit=True)


def inject_trace_context(carrier: dict):
    """Inject current trace context into a carrier dictionary.
    
    This is used to propagate trace context across service boundaries
    via WebSocket messages.
    
    Args:
        carrier: Dictionary to inject trace context into (will add 'traceparent' field)
    """
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    propagator = TraceContextTextMapPropagator()
    propagator.inject(carrier)


def extract_trace_context(carrier: dict):
    """Extract trace context from a carrier dictionary and set as current context.
    
    This is used to continue a trace that was started in another service.
    
    Args:
        carrier: Dictionary with trace context (should have 'traceparent' field)
        
    Returns:
        Context token that should be detached when done
    """
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    from opentelemetry import context
    
    propagator = TraceContextTextMapPropagator()
    ctx = propagator.extract(carrier)
    return context.attach(ctx)


# Custom metrics for Raspberry Pi client
def create_client_metrics():
    """Create custom metrics for client monitoring.
    
    Returns:
        Dictionary of metric instruments
    """
    meter = get_meter("raspberry-pi-client")
    
    return {
        # Counter: Number of wake word detections
        "wake_word_detections": meter.create_counter(
            name="wake_word_detections_total",
            description="Total number of wake word detections",
            unit="1",
        ),
        
        # Counter: Number of conversations started
        "conversations_started": meter.create_counter(
            name="conversations_started_total",
            description="Total number of conversations started",
            unit="1",
        ),
        
        # Counter: Number of conversations completed
        "conversations_completed": meter.create_counter(
            name="conversations_completed_total",
            description="Total number of conversations completed",
            unit="1",
        ),
        
        # Counter: Number of errors
        "errors": meter.create_counter(
            name="errors_total",
            description="Total number of errors",
            unit="1",
        ),
        
        # Histogram: Conversation duration
        "conversation_duration": meter.create_histogram(
            name="conversation_duration_seconds",
            description="Duration of conversations in seconds",
            unit="s",
        ),
        
        # Histogram: Audio latency
        "audio_latency": meter.create_histogram(
            name="audio_latency_milliseconds",
            description="Audio processing latency in milliseconds",
            unit="ms",
        ),
        
        # UpDownCounter: Connection status
        "connection_status": meter.create_up_down_counter(
            name="connection_status",
            description="Connection status (1=connected, 0=disconnected)",
            unit="1",
        ),
    }
