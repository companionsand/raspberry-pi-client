"""
Web Dashboard Server for Kin AI.

Provides:
- FastAPI web server with WebSocket streaming
- Real-time audio, scalar, and text signal broadcasting
- mDNS registration for .local hostname access
- Static file serving for the dashboard UI
"""

import asyncio
import base64
import json
import logging
import socket
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from zeroconf import ServiceInfo, Zeroconf

from lib.config import Config
from lib.signals import ScalarSignal, TextSignal

if TYPE_CHECKING:
    from lib.engine import KinEngine

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for broadcasting."""
    
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total: {len(self.active_connections)}")
    
    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection."""
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict) -> None:
        """Broadcast a message to all connected clients."""
        if not self.active_connections:
            return
        
        data = json.dumps(message)
        disconnected = []
        
        async with self._lock:
            connections = list(self.active_connections)
        
        for connection in connections:
            try:
                await connection.send_text(data)
            except Exception:
                disconnected.append(connection)
        
        # Clean up disconnected clients
        if disconnected:
            async with self._lock:
                for conn in disconnected:
                    if conn in self.active_connections:
                        self.active_connections.remove(conn)


class WebDashboardServer:
    """
    Web dashboard server for Kin AI visualization.
    
    Streams real-time audio, scalar signals, and text events to browser clients
    via WebSocket. Registers mDNS service for easy discovery on local network.
    
    Usage:
        server = WebDashboardServer(engine)
        await server.start()
    """
    
    # Audio streams available for visualization
    AUDIO_STREAMS = [
        ("aec_input", "Echo-Cancelled Input"),
        ("speaker_loopback", "Speaker Output (Loopback)"),
        ("agent_output", "Agent Output"),
        ("raw_input", "Raw Input (6ch)"),
    ]
    
    def __init__(self, engine: "KinEngine"):
        """
        Initialize the web dashboard server.
        
        Args:
            engine: KinEngine instance to visualize
        """
        self.engine = engine
        self.manager = ConnectionManager()
        self.app = self._create_app()
        self.zeroconf: Optional[Zeroconf] = None
        self.service_info: Optional[ServiceInfo] = None
        self._audio_task: Optional[asyncio.Task] = None
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        
        # Track last audio write times per stream
        self._last_audio_times: dict[str, float] = {}
    
    def _create_app(self) -> FastAPI:
        """Create and configure the FastAPI application."""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            """Handle startup and shutdown."""
            # Startup
            self._running = True
            self._loop = asyncio.get_running_loop()
            self._start_signal_subscriptions()
            self._audio_task = asyncio.create_task(self._audio_broadcast_loop())
            self._register_mdns()
            logger.info(f"Web dashboard available at http://{Config.WEB_HOSTNAME}.local:{Config.WEB_PORT}")
            
            yield
            
            # Shutdown
            self._running = False
            self._loop = None
            if self._audio_task:
                self._audio_task.cancel()
                try:
                    await self._audio_task
                except asyncio.CancelledError:
                    pass
            self._unregister_mdns()
        
        app = FastAPI(
            title="Kin AI Dashboard",
            description="Real-time visualization of Kin AI audio and signals",
            lifespan=lifespan
        )
        
        # Serve static files
        static_dir = Path(__file__).parent / "static"
        if static_dir.exists():
            app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        
        @app.get("/", response_class=HTMLResponse)
        async def get_dashboard():
            """Serve the main dashboard page."""
            index_path = Path(__file__).parent / "static" / "index.html"
            if index_path.exists():
                return index_path.read_text()
            return HTMLResponse(
                content="<html><body><h1>Dashboard not found</h1></body></html>",
                status_code=404
            )
        
        @app.get("/api/streams")
        async def get_streams():
            """Get available audio streams."""
            return {"streams": self.AUDIO_STREAMS}
        
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time data streaming."""
            await self.manager.connect(websocket)
            
            # Send initial configuration
            await websocket.send_text(json.dumps({
                "type": "config",
                "sample_rate": Config.SAMPLE_RATE,
                "streams": self.AUDIO_STREAMS,
            }))
            
            try:
                while True:
                    # Handle incoming messages (commands from client)
                    data = await websocket.receive_text()
                    await self._handle_client_message(data)
            except WebSocketDisconnect:
                await self.manager.disconnect(websocket)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await self.manager.disconnect(websocket)
        
        @app.post("/api/inject-wake-word")
        async def inject_wake_word():
            """Inject a simulated wake word detection."""
            self.engine.inject_wake_word()
            return {"status": "ok", "message": "Wake word injected"}
        
        return app
    
    async def _handle_client_message(self, data: str) -> None:
        """
        Handle incoming WebSocket message from client.
        
        Args:
            data: JSON string message from client
        """
        try:
            message = json.loads(data)
            msg_type = message.get("type")
            
            if msg_type == "inject_wake_word":
                self.engine.inject_wake_word()
            elif msg_type == "ping":
                # Respond to keep-alive pings
                pass
            else:
                logger.debug(f"Unknown client message type: {msg_type}")
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON from client: {data}")
    
    def _start_signal_subscriptions(self) -> None:
        """Subscribe to signals from the engine."""
        # Subscribe to TextSignals
        self.engine.signal_bus.subscribe(
            signal_type=TextSignal,
            callback=self._on_text_signal
        )
        
        # Subscribe to ScalarSignals
        self.engine.signal_bus.subscribe(
            signal_type=ScalarSignal,
            callback=self._on_scalar_signal
        )
        
        # Note: AudioSignals are handled by the audio broadcast loop
        # to batch audio chunks for efficiency
    
    def _on_text_signal(self, signal: TextSignal) -> None:
        """Handle incoming TextSignal (called from SignalBus thread)."""
        if self._loop is None:
            return
        
        message = {
            "type": "text",
            "timestamp": signal.timestamp,
            "category": signal.category,
            "message": signal.message,
            "level": signal.level,
        }
        # Schedule broadcast on the event loop thread (capture message by value with default arg)
        self._loop.call_soon_threadsafe(
            lambda msg=message: asyncio.create_task(self.manager.broadcast(msg))
        )
    
    def _on_scalar_signal(self, signal: ScalarSignal) -> None:
        """Handle incoming ScalarSignal (called from SignalBus thread)."""
        if self._loop is None:
            return
        
        message = {
            "type": "scalar",
            "timestamp": signal.timestamp,
            "name": signal.name,
            "value": signal.value,
        }
        # Schedule broadcast on the event loop thread (capture message by value with default arg)
        self._loop.call_soon_threadsafe(
            lambda msg=message: asyncio.create_task(self.manager.broadcast(msg))
        )
    
    async def _audio_broadcast_loop(self) -> None:
        """
        Continuously broadcast audio chunks to connected clients.
        
        Runs at ~30 FPS, sending recent audio for each active stream.
        """
        interval = 1.0 / 30  # ~30 FPS
        audio_window_seconds = 0.05  # 50ms of audio per update
        
        while self._running:
            try:
                start_time = time.monotonic()
                
                if self.manager.active_connections:
                    # Broadcast audio for each stream
                    for stream_name, _ in self.AUDIO_STREAMS:
                        last_write = self.engine.get_stream_last_write_time(stream_name)
                        
                        # Only send if there's new data
                        if last_write > self._last_audio_times.get(stream_name, 0):
                            self._last_audio_times[stream_name] = last_write
                            
                            # Get recent audio
                            audio = self.engine.get_audio_window(stream_name, audio_window_seconds)
                            
                            if len(audio) > 0:
                                # Convert to bytes and base64 encode for transmission
                                audio_bytes = audio.tobytes()
                                audio_b64 = base64.b64encode(audio_bytes).decode('ascii')
                                
                                await self.manager.broadcast({
                                    "type": "audio",
                                    "stream": stream_name,
                                    "timestamp": time.monotonic(),
                                    "data": audio_b64,
                                    "samples": len(audio),
                                })
                
                # Maintain consistent frame rate
                elapsed = time.monotonic() - start_time
                if elapsed < interval:
                    await asyncio.sleep(interval - elapsed)
                else:
                    await asyncio.sleep(0.001)  # Yield to other tasks
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Audio broadcast error: {e}")
                await asyncio.sleep(0.1)
    
    def _register_mdns(self) -> None:
        """Register mDNS service for .local hostname access."""
        try:
            hostname = Config.WEB_HOSTNAME
            port = Config.WEB_PORT
            
            # Get local IP addresses
            local_ips = self._get_local_ips()
            if not local_ips:
                logger.warning("Could not determine local IP address for mDNS")
                return
            
            self.zeroconf = Zeroconf()
            
            # Create service info
            self.service_info = ServiceInfo(
                "_http._tcp.local.",
                f"{hostname}._http._tcp.local.",
                addresses=[socket.inet_aton(ip) for ip in local_ips],
                port=port,
                properties={
                    "path": "/",
                    "version": "1.0",
                    "name": "Kin AI Dashboard",
                },
                server=f"{hostname}.local.",
            )
            
            self.zeroconf.register_service(self.service_info)
            logger.info(f"mDNS registered: {hostname}.local:{port} ({', '.join(local_ips)})")
            
        except Exception as e:
            logger.error(f"Failed to register mDNS: {e}")
    
    def _unregister_mdns(self) -> None:
        """Unregister mDNS service."""
        try:
            if self.service_info and self.zeroconf:
                self.zeroconf.unregister_service(self.service_info)
            if self.zeroconf:
                self.zeroconf.close()
        except Exception as e:
            logger.warning(f"Error unregistering mDNS: {e}")
    
    def _get_local_ips(self) -> list[str]:
        """
        Get local IP addresses for this machine.
        
        Returns:
            List of local IPv4 addresses (excluding loopback)
        """
        ips = []
        try:
            # Create a dummy socket to find the default route IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                ips.append(s.getsockname()[0])
        except Exception:
            pass
        
        # Fallback: try to get all interfaces
        if not ips:
            try:
                hostname = socket.gethostname()
                for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
                    ip = info[4][0]
                    if not ip.startswith("127."):
                        ips.append(ip)
            except Exception:
                pass
        
        return list(set(ips))  # Remove duplicates
    
    async def start(self) -> None:
        """Start the web dashboard server."""
        import uvicorn
        
        config = uvicorn.Config(
            self.app,
            host="0.0.0.0",
            port=Config.WEB_PORT,
            log_level="warning",  # Reduce uvicorn logging noise
        )
        server = uvicorn.Server(config)
        await server.serve()

