# Music Player Module

Internet radio streaming with genre-based station selection.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    OVERNIGHT JOB (cron)                      │
├─────────────────────────────────────────────────────────────┤
│  scripts/update_radio_cache.py                              │
│  - Queries Radio Browser API for stations by genre          │
│  - Saves to ~/.kin_radio_cache.json                         │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    RUNTIME (user request)                    │
├─────────────────────────────────────────────────────────────┤
│  User: "Play jazz"                                          │
│       ↓                                                     │
│  StationRegistry.find_by_genre("jazz")                      │
│       ↓                                                     │
│  1. Check local cache (fast, no network)                    │
│  2. Fall back to curated list if cache miss                 │
│       ↓                                                     │
│  MusicPlayer.start(station.url)                             │
│       ↓                                                     │
│  mpv subprocess plays stream                                │
└─────────────────────────────────────────────────────────────┘
```

## Components

| File | Purpose |
|------|---------|
| `player.py` | MusicPlayer class - plays streams via mpv |
| `stations.py` | StationRegistry - finds stations from cache/curated |
| `radio_browser.py` | Radio Browser API client |
| `test_radio.py` | CLI test script |

## Quick Start

### 1. Install mpv

```bash
# Mac
brew install mpv

# Linux (Raspberry Pi)
sudo apt install mpv
```

### 2. Initialize Cache (first time)

```bash
cd /path/to/raspberry-pi-client
python scripts/update_radio_cache.py
```

This creates `~/.kin_radio_cache.json` with stations for each genre.

### 3. Test Playback

```bash
cd lib/music

# Play by genre
python test_radio.py "jazz"
python test_radio.py "rock music"
python test_radio.py "classical"

# Play default (any music)
python test_radio.py

# List cache stats
python test_radio.py --stats
```

## Usage in Code

```python
from lib.music.player import MusicPlayer

player = MusicPlayer()

# Play by genre
player.play_genre("jazz")

# Play by search query
player.play_query("Frank Sinatra")

# Play default (any music)
player.play_default()

# Stop playback
player.stop()

# Check status
if player.is_active():
    station = player.get_current_station()
    print(f"Playing: {station.name}")
```

## Cache Management

### Manual Refresh

```bash
python scripts/update_radio_cache.py
```

### Cron Setup (Raspberry Pi)

```bash
# Edit crontab
crontab -e

# Add line (runs at 3am daily)
0 3 * * * /home/kin/venv/bin/python /home/kin/raspberry-pi-client/scripts/update_radio_cache.py
```

### Cache Location

- File: `~/.kin_radio_cache.json`
- Contents: Stations grouped by genre (jazz, rock, classical, etc.)
- Source: [Radio Browser API](https://api.radio-browser.info/)

## Supported Genres

From cache (Radio Browser API):
- jazz, classical, rock, pop, country, blues, electronic, ambient, folk, soul

From curated fallback:
- jazz, classical, rock, ambient, electronic, default

Genre aliases (mapped automatically):
- chill → ambient
- lounge → jazz
- techno → electronic
- symphony → classical

## Curated Fallback Stations

If cache is empty or search fails, these reliable stations are used:

| Genre | Station | URL |
|-------|---------|-----|
| default | SomaFM Groove Salad | ice2.somafm.com/groovesalad-128-mp3 |
| jazz | SomaFM Secret Agent | ice2.somafm.com/secretagent-128-mp3 |
| ambient | SomaFM Deep Space One | ice1.somafm.com/deepspaceone-128-mp3 |
| rock | Radio Bob | streams.radiobob.de/bob-live/mp3-192 |

## Troubleshooting

**mpv not found:**
```bash
brew install mpv  # Mac
sudo apt install mpv  # Linux
```

**No stations playing:**
```bash
# Refresh cache
python scripts/update_radio_cache.py

# Check cache stats
python lib/music/test_radio.py --stats
```

**Stream fails:**
- Check internet connection
- Try different genre (some streams may be geo-restricted)
- Curated fallback should always work
