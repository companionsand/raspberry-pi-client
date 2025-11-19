# Raspberry Pi 5 + ReSpeaker Quick Setup

> Run the Kin AI Pi client end to end in ~15 minutes.

---

## 1. Hardware Checklist

- Raspberry Pi 5 (2GB min, 4GB recommended) + 5V/3A USB‑C PSU  
- ReSpeaker 4 Mic Array v2.0 (USB) + powered speaker on the ReSpeaker 3.5 mm jack  
- 32 GB micro‑SD (Raspberry Pi OS Lite 64‑bit)  
- Ethernet or strong Wi‑Fi, heatsink/fan recommended  

---

## 2. Base System

```bash
# flash OS with Pi Imager (enable SSH, set hostname kin-ai, user kin)
ssh kin@kin-ai.local        # default password you set

# keep SSH alive
cat <<'EOF' >> ~/.ssh/config
Host kin-ai.local
    ServerAliveInterval 60
    ServerAliveCountMax 3
    TCPKeepAlive yes
EOF

# basic packages
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-venv python3-pip alsa-utils ffmpeg git
```

Disable PipeWire/PulseAudio (we rely on ALSA + ReSpeaker hardware AEC):

```bash
systemctl --user stop pipewire pipewire-pulse wireplumber
systemctl --user disable pipewire pipewire-pulse wireplumber
systemctl --user mask pipewire pipewire-pulse wireplumber
```

---

## 3. Audio (ALSA‑only, ReSpeaker as default)

```bash
arecord -l   # confirm ReSpeaker (usually card 2)
aplay -l

cat <<'EOF' | sudo tee /etc/asound.conf
pcm.!default {
    type asym
    playback.pcm "plughw:2,0"
    capture.pcm  "plughw:2,0"
}
ctl.!default {
    type hw
    card 2
}
EOF
```

Test:

```bash
arecord -d 3 -f S16_LE -r 16000 -c 1 test.wav
aplay test.wav
```

You should hear yourself from the speaker plugged into the ReSpeaker.

---

## 4. App Install

```bash
# one‑time
python3 -m venv ~/kin-ai-env
source ~/kin-ai-env/bin/activate
mkdir -p ~/kin-ai-prototype
cd ~/kin-ai-prototype

# copy code from laptop
scp pi-client/new_main.py kin@kin-ai.local:~/kin-ai-prototype/main.py
scp pi-client/requirements.txt kin@kin-ai.local:~/kin-ai-prototype/requirements.txt

pip install -r requirements.txt
```

`.env` template:

```bash
cat <<'EOF' > ~/kin-ai-prototype/.env
DEVICE_ID=...
SUPABASE_URL=...
SUPABASE_ANON_KEY=...
EMAIL=...
PASSWORD=...
CONVERSATION_ORCHESTRATOR_URL=wss://conversation-orchestrator.onrender.com/ws
ELEVENLABS_API_KEY=...
PICOVOICE_ACCESS_KEY=...
WAKE_WORD=porcupine
LED_ENABLED=true
EOF
chmod 600 ~/kin-ai-prototype/.env
```

Run it:

```bash
source ~/kin-ai-env/bin/activate
cd ~/kin-ai-prototype
python main.py
```

Expected logs: Supabase auth success → ALSA verification → “Ready! Say ‘porcupine’…”.

---

## 5. Deploy as Service (optional)

```bash
cat <<'EOF' | sudo tee /etc/systemd/system/kin-client.service
[Unit]
Description=Kin AI Client
After=network-online.target sound.target

[Service]
User=kin
WorkingDirectory=/home/kin/kin-ai-prototype
Environment="PATH=/home/kin/kin-ai-env/bin"
ExecStart=/home/kin/kin-ai-env/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now kin-client
```

Useful commands:

```bash
sudo systemctl status kin-client
sudo journalctl -u kin-client -f
```

---

## 6. Troubleshooting Cheat Sheet

| Symptom | Fix |
| ------- | --- |
| `arecord` silent / -91 dB | Re-check `/etc/asound.conf`, run `arecord -D plughw:2,0 ...`, ensure ReSpeaker is card 2. |
| `python -c "import sounddevice"` hangs | PipeWire still running → re-run the disable commands above. |
| Wake word ignored | Run `arecord -d 3 test.wav`; if silent, check mic gain (`amixer -c 2`). |
| No speaker audio | Ensure speaker is on ReSpeaker jack, run `aplay test.wav`, check `amixer -c 2 set Speaker 80%`. |
| LEDs off | `pip install pixel-ring`, or set `LED_ENABLED=false`. |
| Ctrl+C not working | Latest `main.py` handles SIGINT; ensure you redeployed it. |

---

## 7. Quick Reference

```bash
# Audio devices
arecord -l
aplay -l

# Temp / throttling
vcgencmd measure_temp
vcgencmd get_throttled

# Logs (manual run)
python main.py

# Logs (service)
sudo journalctl -u kin-client -f

# Restart service
sudo systemctl restart kin-client
```

That’s it—once `python main.py` shows “Ready! Say ‘porcupine’…”, you’re good to test wake word + conversation.***

