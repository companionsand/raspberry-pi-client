#!/usr/bin/env python3
"""
MR60FDA1 Radar Sensor - Real-Time Dashboard

A beautiful web-based visualization for investor demos.
Shows presence detection, movement intensity, and fall alerts in real-time.

Usage:
    python scripts/radar_dashboard.py
    python scripts/radar_dashboard.py --port /dev/ttyUSB0

Then open: http://localhost:8080 (or http://<pi-ip>:8080 from another device)
"""

import sys
import os
import json
import time
import asyncio
import argparse
import threading
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Optional
import socket

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.radar_sensor import MR60FDA1Sensor, PresenceState, FallState, RadarReading


# Global state for sharing between threads
class DashboardState:
    def __init__(self):
        self.current_reading: Optional[RadarReading] = None
        self.events: list = []  # Recent events for timeline
        self.start_time = datetime.now()
        self.lock = threading.Lock()
    
    def update_reading(self, reading: RadarReading):
        with self.lock:
            self.current_reading = reading
    
    def add_event(self, event_type: str, details: str):
        with self.lock:
            self.events.append({
                "time": datetime.now().isoformat(),
                "type": event_type,
                "details": details
            })
            # Keep last 50 events
            if len(self.events) > 50:
                self.events = self.events[-50:]
    
    def to_json(self) -> str:
        with self.lock:
            if self.current_reading is None:
                return json.dumps({"status": "waiting"})
            
            return json.dumps({
                "status": "active",
                "presence": self.current_reading.presence.name,
                "presence_value": self.current_reading.presence.value,
                "movement_intensity": self.current_reading.movement_intensity,
                "fall_state": self.current_reading.fall_state.name,
                "fall_value": self.current_reading.fall_state.value,
                "stationary_duration": self.current_reading.stationary_duration,
                "timestamp": self.current_reading.timestamp.isoformat(),
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
                "recent_events": self.events[-10:]  # Last 10 for display
            })


# Global state instance
dashboard_state = DashboardState()


# HTML Dashboard (embedded for simplicity)
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kin Radar - Presence & Fall Detection</title>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0a0f;
            --bg-secondary: #12121a;
            --bg-card: #1a1a24;
            --accent-cyan: #00d4ff;
            --accent-green: #00ff88;
            --accent-yellow: #ffcc00;
            --accent-red: #ff4444;
            --accent-purple: #a855f7;
            --text-primary: #ffffff;
            --text-secondary: #8888aa;
            --border-color: #2a2a3a;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Space Grotesk', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
        }
        
        /* Animated background gradient */
        .bg-gradient {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: 
                radial-gradient(ellipse at 20% 20%, rgba(0, 212, 255, 0.1) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 80%, rgba(168, 85, 247, 0.1) 0%, transparent 50%),
                var(--bg-primary);
            z-index: -1;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        header {
            text-align: center;
            margin-bottom: 3rem;
        }
        
        .logo {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        
        .subtitle {
            color: var(--text-secondary);
            font-size: 1.1rem;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            position: relative;
            overflow: hidden;
        }
        
        .card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple));
        }
        
        .card-title {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-secondary);
            margin-bottom: 1rem;
        }
        
        /* Presence Display */
        .presence-display {
            text-align: center;
            padding: 2rem 0;
        }
        
        .presence-icon {
            width: 120px;
            height: 120px;
            border-radius: 50%;
            margin: 0 auto 1.5rem;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 3rem;
            transition: all 0.5s ease;
            box-shadow: 0 0 40px rgba(0, 212, 255, 0.3);
        }
        
        .presence-icon.none {
            background: linear-gradient(135deg, #2a2a3a, #1a1a24);
            box-shadow: none;
        }
        
        .presence-icon.stationary {
            background: linear-gradient(135deg, var(--accent-yellow), #ff9900);
            animation: pulse-yellow 2s ease-in-out infinite;
        }
        
        .presence-icon.moving {
            background: linear-gradient(135deg, var(--accent-green), #00cc66);
            animation: pulse-green 1s ease-in-out infinite;
        }
        
        @keyframes pulse-yellow {
            0%, 100% { box-shadow: 0 0 40px rgba(255, 204, 0, 0.4); }
            50% { box-shadow: 0 0 60px rgba(255, 204, 0, 0.6); }
        }
        
        @keyframes pulse-green {
            0%, 100% { box-shadow: 0 0 40px rgba(0, 255, 136, 0.4); }
            50% { box-shadow: 0 0 60px rgba(0, 255, 136, 0.6); }
        }
        
        .presence-label {
            font-size: 1.5rem;
            font-weight: 600;
        }
        
        /* Movement Bar */
        .movement-container {
            margin-top: 1rem;
        }
        
        .movement-bar-bg {
            height: 12px;
            background: var(--bg-secondary);
            border-radius: 6px;
            overflow: hidden;
        }
        
        .movement-bar {
            height: 100%;
            background: linear-gradient(90deg, var(--accent-cyan), var(--accent-green));
            border-radius: 6px;
            transition: width 0.3s ease;
        }
        
        .movement-value {
            text-align: right;
            margin-top: 0.5rem;
            font-size: 2rem;
            font-weight: 700;
            color: var(--accent-cyan);
        }
        
        /* Fall Alert */
        .fall-alert {
            text-align: center;
            padding: 2rem;
        }
        
        .fall-status {
            font-size: 1.3rem;
            font-weight: 600;
            padding: 1rem 2rem;
            border-radius: 12px;
            display: inline-block;
        }
        
        .fall-status.normal {
            background: rgba(0, 255, 136, 0.1);
            border: 1px solid var(--accent-green);
            color: var(--accent-green);
        }
        
        .fall-status.suspected {
            background: rgba(255, 204, 0, 0.2);
            border: 2px solid var(--accent-yellow);
            color: var(--accent-yellow);
            animation: blink 1s ease-in-out infinite;
        }
        
        .fall-status.confirmed {
            background: rgba(255, 68, 68, 0.3);
            border: 3px solid var(--accent-red);
            color: white;
            animation: alert 0.5s ease-in-out infinite;
        }
        
        @keyframes blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }
        
        @keyframes alert {
            0%, 100% { transform: scale(1); background: rgba(255, 68, 68, 0.3); }
            50% { transform: scale(1.02); background: rgba(255, 68, 68, 0.5); }
        }
        
        /* Stationary Timer */
        .timer-display {
            text-align: center;
            padding: 1.5rem 0;
        }
        
        .timer-value {
            font-size: 3rem;
            font-weight: 700;
            font-variant-numeric: tabular-nums;
            color: var(--accent-purple);
        }
        
        .timer-label {
            color: var(--text-secondary);
            margin-top: 0.5rem;
        }
        
        /* Event Timeline */
        .timeline {
            max-height: 300px;
            overflow-y: auto;
        }
        
        .event-item {
            display: flex;
            align-items: center;
            padding: 0.75rem;
            border-bottom: 1px solid var(--border-color);
            font-size: 0.9rem;
        }
        
        .event-item:last-child {
            border-bottom: none;
        }
        
        .event-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 1rem;
            flex-shrink: 0;
        }
        
        .event-dot.presence { background: var(--accent-cyan); }
        .event-dot.movement { background: var(--accent-green); }
        .event-dot.fall { background: var(--accent-red); }
        
        .event-time {
            color: var(--text-secondary);
            font-size: 0.8rem;
            margin-left: auto;
            padding-left: 1rem;
        }
        
        /* Status Bar */
        .status-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 1.5rem;
            background: var(--bg-card);
            border-radius: 12px;
            border: 1px solid var(--border-color);
        }
        
        .status-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--accent-green);
            animation: pulse-dot 2s ease-in-out infinite;
        }
        
        @keyframes pulse-dot {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .status-dot.offline {
            background: var(--accent-red);
            animation: none;
        }
        
        /* Radar Visualization */
        .radar-viz {
            position: relative;
            width: 200px;
            height: 200px;
            margin: 0 auto;
        }
        
        .radar-circle {
            position: absolute;
            border: 1px solid var(--border-color);
            border-radius: 50%;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
        }
        
        .radar-circle:nth-child(1) { width: 50px; height: 50px; }
        .radar-circle:nth-child(2) { width: 100px; height: 100px; }
        .radar-circle:nth-child(3) { width: 150px; height: 150px; }
        .radar-circle:nth-child(4) { width: 200px; height: 200px; }
        
        .radar-sweep {
            position: absolute;
            width: 100px;
            height: 2px;
            background: linear-gradient(90deg, var(--accent-cyan), transparent);
            top: 50%;
            left: 50%;
            transform-origin: left center;
            animation: sweep 3s linear infinite;
        }
        
        @keyframes sweep {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        
        .radar-blip {
            position: absolute;
            width: 12px;
            height: 12px;
            background: var(--accent-green);
            border-radius: 50%;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            opacity: 0;
            transition: all 0.5s ease;
        }
        
        .radar-blip.active {
            opacity: 1;
            box-shadow: 0 0 20px var(--accent-green);
        }
        
        .radar-blip.moving {
            animation: blip-pulse 1s ease-in-out infinite;
        }
        
        @keyframes blip-pulse {
            0%, 100% { transform: translate(-50%, -50%) scale(1); }
            50% { transform: translate(-50%, -50%) scale(1.3); }
        }
    </style>
</head>
<body>
    <div class="bg-gradient"></div>
    
    <div class="container">
        <header>
            <h1 class="logo">Kin Radar</h1>
            <p class="subtitle">60GHz mmWave Presence & Fall Detection</p>
        </header>
        
        <div class="grid">
            <!-- Presence Card -->
            <div class="card">
                <div class="card-title">Presence Status</div>
                <div class="presence-display">
                    <div class="presence-icon none" id="presence-icon">
                        <span id="presence-emoji">👤</span>
                    </div>
                    <div class="presence-label" id="presence-label">Waiting...</div>
                </div>
            </div>
            
            <!-- Radar Visualization -->
            <div class="card">
                <div class="card-title">Radar View</div>
                <div class="radar-viz">
                    <div class="radar-circle"></div>
                    <div class="radar-circle"></div>
                    <div class="radar-circle"></div>
                    <div class="radar-circle"></div>
                    <div class="radar-sweep"></div>
                    <div class="radar-blip" id="radar-blip"></div>
                </div>
            </div>
            
            <!-- Fall Detection -->
            <div class="card">
                <div class="card-title">Fall Detection</div>
                <div class="fall-alert">
                    <div class="fall-status normal" id="fall-status">✓ Normal</div>
                </div>
            </div>
        </div>
        
        <div class="grid">
            <!-- Movement Intensity -->
            <div class="card">
                <div class="card-title">Movement Intensity</div>
                <div class="movement-container">
                    <div class="movement-bar-bg">
                        <div class="movement-bar" id="movement-bar" style="width: 0%"></div>
                    </div>
                    <div class="movement-value"><span id="movement-value">0</span>%</div>
                </div>
            </div>
            
            <!-- Stationary Timer -->
            <div class="card">
                <div class="card-title">Stationary Duration</div>
                <div class="timer-display">
                    <div class="timer-value" id="stationary-timer">0:00</div>
                    <div class="timer-label">Time without movement</div>
                </div>
            </div>
            
            <!-- Event Timeline -->
            <div class="card">
                <div class="card-title">Recent Events</div>
                <div class="timeline" id="timeline">
                    <div class="event-item" style="color: var(--text-secondary)">
                        Waiting for events...
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Status Bar -->
        <div class="status-bar">
            <div class="status-item">
                <div class="status-dot" id="connection-dot"></div>
                <span id="connection-status">Connecting...</span>
            </div>
            <div class="status-item">
                <span id="uptime">Uptime: --:--</span>
            </div>
        </div>
    </div>
    
    <script>
        // Poll for updates
        async function fetchData() {
            try {
                const response = await fetch('/api/data');
                const data = await response.json();
                updateDashboard(data);
            } catch (e) {
                setOffline();
            }
        }
        
        function updateDashboard(data) {
            if (data.status === 'waiting') {
                document.getElementById('connection-status').textContent = 'Waiting for sensor...';
                return;
            }
            
            // Connection status
            document.getElementById('connection-dot').classList.remove('offline');
            document.getElementById('connection-status').textContent = 'Connected';
            
            // Presence
            const presenceIcon = document.getElementById('presence-icon');
            const presenceLabel = document.getElementById('presence-label');
            const presenceEmoji = document.getElementById('presence-emoji');
            const radarBlip = document.getElementById('radar-blip');
            
            presenceIcon.className = 'presence-icon ' + data.presence.toLowerCase();
            
            if (data.presence === 'NONE') {
                presenceLabel.textContent = 'No Presence';
                presenceEmoji.textContent = '👤';
                radarBlip.classList.remove('active', 'moving');
            } else if (data.presence === 'STATIONARY') {
                presenceLabel.textContent = 'Stationary';
                presenceEmoji.textContent = '🧍';
                radarBlip.classList.add('active');
                radarBlip.classList.remove('moving');
            } else {
                presenceLabel.textContent = 'Moving';
                presenceEmoji.textContent = '🚶';
                radarBlip.classList.add('active', 'moving');
            }
            
            // Movement
            document.getElementById('movement-bar').style.width = data.movement_intensity + '%';
            document.getElementById('movement-value').textContent = data.movement_intensity;
            
            // Fall status
            const fallStatus = document.getElementById('fall-status');
            fallStatus.className = 'fall-status ' + data.fall_state.toLowerCase();
            if (data.fall_state === 'NORMAL') {
                fallStatus.textContent = '✓ Normal';
            } else if (data.fall_state === 'SUSPECTED') {
                fallStatus.textContent = '⚠️ SUSPECTED FALL';
            } else {
                fallStatus.textContent = '🚨 CONFIRMED FALL';
            }
            
            // Stationary timer
            const mins = Math.floor(data.stationary_duration / 60);
            const secs = data.stationary_duration % 60;
            document.getElementById('stationary-timer').textContent = 
                mins + ':' + secs.toString().padStart(2, '0');
            
            // Uptime
            const uptimeMins = Math.floor(data.uptime_seconds / 60);
            const uptimeSecs = Math.floor(data.uptime_seconds % 60);
            document.getElementById('uptime').textContent = 
                'Uptime: ' + uptimeMins + ':' + uptimeSecs.toString().padStart(2, '0');
            
            // Events
            if (data.recent_events && data.recent_events.length > 0) {
                const timeline = document.getElementById('timeline');
                timeline.innerHTML = data.recent_events.reverse().map(event => {
                    const time = new Date(event.time).toLocaleTimeString();
                    return `
                        <div class="event-item">
                            <div class="event-dot ${event.type}"></div>
                            <span>${event.details}</span>
                            <span class="event-time">${time}</span>
                        </div>
                    `;
                }).join('');
            }
        }
        
        function setOffline() {
            document.getElementById('connection-dot').classList.add('offline');
            document.getElementById('connection-status').textContent = 'Disconnected';
        }
        
        // Poll every 200ms for smooth updates
        setInterval(fetchData, 200);
        fetchData();
    </script>
</body>
</html>
"""


class DashboardHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler for the dashboard."""
    
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode())
        elif self.path == '/api/data':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(dashboard_state.to_json().encode())
        else:
            self.send_error(404)
    
    def log_message(self, format, *args):
        # Suppress HTTP logs for cleaner output
        pass


def get_local_ip():
    """Get local IP address for display."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "localhost"


def run_sensor(port: str = None):
    """Run the radar sensor in a thread."""
    
    def on_presence(presence: PresenceState, intensity: int):
        dashboard_state.add_event("presence", f"Presence: {presence.name} ({intensity}%)")
    
    def on_fall(fall_state: FallState):
        severity = "SUSPECTED" if fall_state == FallState.SUSPECTED else "CONFIRMED"
        dashboard_state.add_event("fall", f"⚠️ Fall {severity}!")
    
    sensor = MR60FDA1Sensor(
        port=port,
        on_presence=on_presence,
        on_fall=on_fall
    )
    
    if not sensor.start():
        print("✗ Failed to start radar sensor")
        return
    
    print(f"✓ Radar sensor started on {sensor.port}")
    
    try:
        while True:
            time.sleep(0.1)
            reading = sensor.get_reading()
            dashboard_state.update_reading(reading)
    except:
        pass
    finally:
        sensor.stop()


def main():
    parser = argparse.ArgumentParser(description='MR60FDA1 Radar Dashboard')
    parser.add_argument('--port', '-p', type=str, default=None,
                        help='Serial port (e.g., /dev/ttyUSB0)')
    parser.add_argument('--http-port', type=int, default=8080,
                        help='HTTP server port (default: 8080)')
    args = parser.parse_args()
    
    # Start sensor thread
    sensor_thread = threading.Thread(target=run_sensor, args=(args.port,), daemon=True)
    sensor_thread.start()
    
    # Start HTTP server
    local_ip = get_local_ip()
    server = HTTPServer(('0.0.0.0', args.http_port), DashboardHandler)
    
    print(f"\n{'='*60}")
    print(f"  🌐 Kin Radar Dashboard")
    print(f"{'='*60}")
    print(f"\n  Open in browser:")
    print(f"    Local:   http://localhost:{args.http_port}")
    print(f"    Network: http://{local_ip}:{args.http_port}")
    print(f"\n  Press Ctrl+C to stop\n")
    print(f"{'='*60}\n")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n✓ Dashboard stopped")
        server.shutdown()


if __name__ == "__main__":
    main()

