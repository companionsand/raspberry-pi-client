# Telemetry Reference

Quick reference for all telemetry tracked in the Raspberry Pi Client.

## Metrics

### Counters

| Metric Name | Description | Attributes |
|-------------|-------------|------------|
| `wake_word_detections_total` | Number of wake word detections | `device_id`, `wake_word` |
| `conversations_started_total` | Number of conversations started | `device_id`, `user_id`, `agent_id` |
| `conversations_completed_total` | Number of conversations completed | `device_id`, `user_id`, `agent_id` |
| `errors_total` | Number of errors occurred | `device_id`, `error_type`, `error_message` |

### Histograms

| Metric Name | Description | Unit | Attributes |
|-------------|-------------|------|------------|
| `conversation_duration_seconds` | Duration of conversations | seconds | `device_id`, `user_id` |

### UpDownCounters

| Metric Name | Description | Values | Attributes |
|-------------|-------------|--------|------------|
| `connection_status` | Connection status | 1=connected, 0/−1=disconnected | `device_id` |

## Events

| Event Name | When Triggered | Attributes |
|------------|----------------|------------|
| `wake_word_detected` | Wake word is detected | `device_id`, `wake_word` |
| `conversation_started` | Conversation begins | `device_id`, `user_id`, `agent_id` |
| `conversation_completed` | Conversation ends | `device_id`, `user_id`, `duration` |
| `orchestrator_connected` | Connected to orchestrator | `device_id` |
| `orchestrator_disconnected` | Disconnected from orchestrator | `device_id` |
| `error_occurred` | Error happens | `device_id`, `error_type`, `error_message` |

## Configuration

### Enable/Disable Telemetry
```bash
# In .env
OTEL_ENABLED=true  # or false
OTEL_EXPORTER_ENDPOINT=http://localhost:4318
ENV=production
```

## API Usage

### Recording Metrics

```python
# Counter
self.metrics["wake_word_detections"].add(1, {
    "device_id": Config.DEVICE_ID
})

# Histogram
self.metrics["conversation_duration"].record(duration, {
    "device_id": Config.DEVICE_ID
})

# UpDownCounter  
self.metrics["connection_status"].add(1, {  # +1 for connect, -1 for disconnect
    "device_id": Config.DEVICE_ID
})
```

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

See `OTEL_SETUP.md` for complete setup and troubleshooting guide.

