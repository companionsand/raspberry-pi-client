# Telemetry Reference

Quick reference for all telemetry tracked in the Raspberry Pi Client.

## Current Telemetry Implementation

The Raspberry Pi Client currently implements **traces, spans, and logs** via OpenTelemetry. Metrics infrastructure exists but is **not actively used** except for limited wake word detector metrics.

### Active Telemetry

- **Traces**: Conversation-level root traces with distributed tracing support
- **Spans**: Detailed spans for operations (conversation handling, audio processing, etc.)
- **Logs**: Structured logging with OTEL logger, stdout/stderr redirection
- **Span Events**: Event annotations on spans (e.g., `orchestrator_connected`, `wake_word_detected`)

### Metrics (Limited Implementation)

Metrics infrastructure is set up but only partially used:

- **Wake Word Detector**: Uses `scribe_verifications_total` counter for Scribe v2 verification attempts
- **Other Metrics**: Not currently implemented (planned for future)

**Note**: The README states "no metrics" to reflect that comprehensive metrics are not yet implemented. The metrics infrastructure exists for future expansion.

## Span Events

Span events are used to annotate traces with important occurrences:

| Event Name | When Triggered | Attributes |
|------------|----------------|------------|
| `orchestrator_connected` | Connected to orchestrator | `device_id` |
| `orchestrator_disconnected` | Disconnected from orchestrator | `device_id` |
| `wake_word_detected` | Wake word is detected | `device_id`, `wake_word` (via structured logging) |
| `conversation_started` | Conversation begins | `device_id`, `user_id`, `agent_id` (via structured logging) |
| `conversation_completed` | Conversation ends | `device_id`, `user_id`, `duration` (via structured logging) |
| `error_occurred` | Error happens | `device_id`, `error_type`, `error_message` (via structured logging) |

**Note**: Most events are tracked via structured logging rather than explicit span events. The telemetry system captures these through the OTEL logger.

## Configuration

### Enable/Disable Telemetry
```bash
# In .env
OTEL_ENABLED=true  # or false
OTEL_EXPORTER_ENDPOINT=http://localhost:4318
ENV=production
```

## API Usage

### Adding Span Events

```python
if TELEMETRY_AVAILABLE:
    add_span_event("event_name",
                   device_id=Config.DEVICE_ID,
                   custom_attr="value")
```

## Monitoring

### Check Telemetry Status
```bash
# Collector health
curl http://localhost:13133/

# Collector logs
sudo journalctl -u otelcol -f

# Client telemetry logs
sudo journalctl -u agent-launcher -f | grep -i "telemetry\|otel"
```

## Trace Context Propagation

The telemetry system supports distributed tracing:

- **Reactive conversations**: Trace context is created locally and propagated to orchestrator via WebSocket messages
- **Proactive conversations**: Trace context is extracted from `start_conversation` message and used for the conversation
- This enables end-to-end distributed tracing across services

## Future Enhancements

Comprehensive metrics are planned for future implementation:
- Wake word detection counters
- Conversation duration histograms
- Connection status up/down counters
- Error rate tracking

