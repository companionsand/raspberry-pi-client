"""OpenTelemetry integration for observability"""

from .telemetry import (
    setup_telemetry,
    get_tracer,
    get_logger,
    add_span_attributes,
    add_span_event,
    create_span,
    create_conversation_trace,
    inject_trace_context,
    extract_trace_context,
    record_exception,
)

from .stdout_redirect import (
    setup_stdout_redirect,
    cleanup_stdout_redirect,
)

__all__ = [
    "setup_telemetry",
    "get_tracer",
    "get_logger",
    "add_span_attributes",
    "add_span_event",
    "create_span",
    "create_conversation_trace",
    "inject_trace_context",
    "extract_trace_context",
    "record_exception",
    "setup_stdout_redirect",
    "cleanup_stdout_redirect",
]

