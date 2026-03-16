# Clients

External API client wrappers used by the tool implementations in `../tools/`.

## Files

| File | Used By Tool | Description |
|------|-------------|-------------|
| `weather_client.py` | `weather` | Fetches weather forecast data from external weather API |
| `kasa_lighting_client.py` | `kasa_lighting` | Controls TP-Link Kasa smart lights via their API |
| `calendar_client.py` | `calendar_data` | Reads/writes Google Calendar events via Google Calendar API |
| `web_search_client.py` | `google_search` | Performs web searches and returns structured results |

## Relationship to Tools

Tools in `../tools/` contain the business logic and argument validation. These clients handle the raw HTTP/API communication:

```
tools/<tool>.py  →  clients/<tool>_client.py  →  External API
```

The separation keeps tool logic (argument normalization, response formatting) decoupled from API transport details (authentication, request formatting, retries).

## Adding a New Client

1. Create `<name>_client.py` with functions the corresponding tool will call
2. Handle API authentication, request formatting, and error handling within the client
3. Import and use the client from the corresponding tool in `../tools/`
