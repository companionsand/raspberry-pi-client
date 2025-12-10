"""
Setup HTTP Server

Provides web interface for WiFi configuration using asyncio HTTP server.
"""

import asyncio
import json
import logging
import subprocess
from http.server import BaseHTTPRequestHandler
from socketserver import TCPServer
from threading import Thread
from typing import Callable, Optional

logger = logging.getLogger(__name__)


HTML_TEMPLATE = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kin Device Setup</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500&family=Instrument+Serif:ital,wght@0,400;1,400&family=Limelight&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        :root {
            --color-primary: #ed572d;
            --color-success: #04602B;
            --color-black: #111111;
            --color-white: #FFFFFF;
            --color-warm-white: #FFF8F3;
        }
        
        body {
            font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background-color: var(--color-warm-white);
            color: var(--color-black);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            font-size: 18px;
            line-height: 1.5;
            filter: contrast(1.02) saturate(1.1);
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }
        
        /* Grainy overlay effect for nostalgic texture */
        body::before {
            content: "";
            position: fixed;
            inset: 0;
            background:
                radial-gradient(circle at 20% 20%, rgba(255,255,255,0.03) 0%, transparent 100%),
                url("https://grainy-gradients.vercel.app/noise.svg");
            mix-blend-mode: overlay;
            opacity: 0.35;
            pointer-events: none;
            z-index: 9999;
        }
        
        .container {
            background: var(--color-white);
            border-radius: 12px;
            border: 1px solid rgba(17, 17, 17, 0.1);
            padding: 40px;
            max-width: 500px;
            width: 100%;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            position: relative;
            z-index: 1;
        }
        
        h1 { 
            font-family: 'Instrument Serif', serif;
            color: var(--color-black);
            margin-bottom: 8px;
            font-size: 28px;
            line-height: 1.2;
            text-transform: uppercase;
            letter-spacing: 0.02em;
            font-weight: 400;
        }
        
        .subtitle { 
            font-family: 'IBM Plex Sans', sans-serif;
            color: var(--color-black);
            opacity: 0.7;
            margin-bottom: 32px;
            font-size: 18px;
            line-height: 1.5;
        }
        
        .form-group { margin-bottom: 20px; }
        
        label { 
            display: block;
            margin-bottom: 8px;
            color: var(--color-black);
            font-weight: 500;
            font-size: 14px;
        }
        
        select, input {
            width: 100%;
            padding: 10px 12px;
            border: 1px solid rgba(17, 17, 17, 0.1);
            border-radius: 8px;
            font-size: 18px;
            font-family: 'IBM Plex Sans', sans-serif;
            background-color: var(--color-white);
            color: var(--color-black);
            transition: all 150ms ease;
        }
        
        select:focus, input:focus {
            outline: none;
            border-color: var(--color-primary);
            box-shadow: 0 0 0 3px rgba(237, 87, 45, 0.15);
        }
        
        button {
            width: 100%;
            padding: 12px 20px;
            background: var(--color-primary);
            color: var(--color-white);
            border: none;
            border-radius: 12px;
            font-size: 18px;
            font-weight: 500;
            font-family: 'IBM Plex Sans', sans-serif;
            cursor: pointer;
            transition: all 150ms ease;
        }
        
        button:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(237, 87, 45, 0.3);
        }
        
        button:active:not(:disabled) {
            transform: translateY(0);
        }
        
        button:disabled { 
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .message { 
            padding: 12px 16px;
            border-radius: 8px;
            margin-bottom: 20px;
            white-space: pre-line;
            font-size: 16px;
        }
        
        .success { 
            background-color: rgba(4, 96, 43, 0.1);
            color: var(--color-success);
            border: 1px solid rgba(4, 96, 43, 0.2);
        }
        
        .error { 
            background-color: rgba(237, 87, 45, 0.1);
            color: var(--color-primary);
            border: 1px solid rgba(237, 87, 45, 0.2);
        }
        
        .info { 
            background-color: rgba(17, 17, 17, 0.05);
            color: var(--color-black);
            border: 1px solid rgba(17, 17, 17, 0.1);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Kin Device Setup</h1>
        <p class="subtitle">Configure your WiFi network to get started</p>
        <div id="message"></div>
        <form id="setupForm">
            <div class="form-group">
                <label for="ssid">WiFi Network</label>
                <select id="ssid" name="ssid" required>
                    <option value="">Scanning...</option>
                </select>
            </div>
            <div class="form-group">
                <label for="password">WiFi Password</label>
                <input type="password" id="password" name="password" placeholder="Leave blank if open">
            </div>
            <div class="form-group">
                <label for="pairingCode">Pairing Code (4 digits)</label>
                <input type="text" id="pairingCode" name="pairingCode" pattern="[0-9]{4}" maxlength="4" placeholder="1234" required>
            </div>
            <button type="submit" id="submitBtn">Connect</button>
        </form>
    </div>
    <script>
        const form = document.getElementById('setupForm');
        const ssidSelect = document.getElementById('ssid');
        const messageDiv = document.getElementById('message');
        const submitBtn = document.getElementById('submitBtn');
        
        function showMessage(text, type) {
            messageDiv.innerHTML = '<div class="message ' + type + '">' + text + '</div>';
        }
        
        function clearMessage() {
            messageDiv.innerHTML = '';
        }
        
        fetch('/networks')
            .then(r => r.json())
            .then(data => {
                ssidSelect.innerHTML = '<option value="">Select network...</option>';
                data.networks.forEach(n => {
                    const opt = document.createElement('option');
                    opt.value = n.ssid;
                    opt.textContent = n.ssid + (n.encrypted ? ' 🔒' : '');
                    ssidSelect.appendChild(opt);
                });
            })
            .catch(e => showMessage('Failed to load networks: ' + e.message, 'error'));
        
        let statusPollInterval = null;
        
        function pollStatus() {
            fetch('/status')
                .then(r => r.json())
                .then(data => {
                    if (data.status === 'waiting') {
                        showMessage(data.message, 'info');
                    } else if (data.status === 'connecting') {
                        showMessage('📶 ' + data.message, 'info');
                    } else if (data.status === 'authenticating') {
                        showMessage('🔐 ' + data.message, 'info');
                    } else if (data.status === 'success') {
                        showMessage('✓ ' + data.message, 'success');
                        clearInterval(statusPollInterval);
                        submitBtn.disabled = true;
                        submitBtn.textContent = 'Setup Complete!';
                    } else if (data.status === 'error') {
                        showMessage('✗ ' + data.message + (data.error ? '\\n' + data.error : ''), 'error');
                        clearInterval(statusPollInterval);
                        submitBtn.disabled = false;
                        submitBtn.textContent = 'Connect';
                    }
                })
                .catch(e => console.error('Status poll failed:', e));
        }
        
        form.onsubmit = async (e) => {
            e.preventDefault();
            clearMessage();  // Clear any old error messages
            submitBtn.disabled = true;
            submitBtn.textContent = 'Submitting...';
            
            try {
                const r = await fetch('/configure', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        ssid: form.ssid.value,
                        password: form.password.value,
                        pairing_code: form.pairingCode.value
                    })
                });
                const result = await r.json();
                if (result.success) {
                    showMessage('Configuration received. Connecting to WiFi...', 'success');
                    // Start polling for status updates
                    statusPollInterval = setInterval(pollStatus, 2000);
                    pollStatus(); // Poll immediately
                } else {
                    showMessage('Error: ' + (result.error || 'Failed'), 'error');
                    submitBtn.disabled = false;
                    submitBtn.textContent = 'Connect';
                }
            } catch(e) {
                showMessage('Error: ' + e.message, 'error');
                submitBtn.disabled = false;
                submitBtn.textContent = 'Connect';
            }
        };
    </script>
</body>
</html>
'''


class SetupHTTPServer:
    """HTTP server for WiFi setup interface"""
    
    def __init__(self, port: int = 8080, wifi_interface: str = "wlan0", ap_ssid: str = "Kin_Setup", ap_password: str = "kinsetup123"):
        self.port = port
        self.wifi_interface = wifi_interface
        self.ap_ssid = ap_ssid
        self.ap_password = ap_password
        self._server: Optional[TCPServer] = None
        self._server_thread: Optional[Thread] = None
        self._config_callback: Optional[Callable] = None
        self._status = "waiting"  # waiting, connecting, authenticating, success, error
        self._status_message = "Waiting for WiFi configuration..."
        self._error_details = None
    
    async def start(self, config_callback: Callable[[str, str, str], bool]):
        """
        Start the HTTP server.
        
        Args:
            config_callback: Callback function to call when user submits configuration.
                           Should accept (ssid, password, pairing_code) and return bool.
        """
        self._config_callback = config_callback
        
        # Create request handler class with reference to this instance
        setup_server = self
        
        class SetupHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                """Override to use our logger"""
                logger.debug(f"[HTTP] {self.address_string()} - {format%args}")
            
            def do_GET(self):
                client_ip = self.address_string()
                logger.info(f"[HTTP] GET {self.path} from {client_ip}")
                
                if self.path == '/' or self.path == '/setup.html':
                    logger.debug(f"[HTTP] Serving setup page to {client_ip}")
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    self.wfile.write(HTML_TEMPLATE.encode())
                elif self.path == '/ap-info':
                    logger.debug(f"[HTTP] Serving AP info to {client_ip}")
                    self._handle_ap_info()
                elif self.path == '/networks':
                    logger.debug(f"[HTTP] Network scan requested by {client_ip}")
                    self._handle_networks()
                elif self.path == '/status':
                    logger.debug(f"[HTTP] Status check from {client_ip}")
                    self._handle_status()
                else:
                    logger.warning(f"[HTTP] 404 Not Found: {self.path} from {client_ip}")
                    self.send_error(404)
            
            def do_POST(self):
                client_ip = self.address_string()
                logger.info(f"[HTTP] POST {self.path} from {client_ip}")
                
                if self.path == '/configure':
                    logger.info(f"[HTTP] Configuration submission from {client_ip}")
                    self._handle_configure()
                else:
                    logger.warning(f"[HTTP] 404 Not Found: {self.path} from {client_ip}")
                    self.send_error(404)
            
            def _send_json(self, status_code, data):
                self.send_response(status_code)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())
            
            def _handle_ap_info(self):
                """Return AP connection info"""
                logger.debug(f"[HTTP] Sending AP info: SSID={setup_server.ap_ssid}")
                ap_data = {
                    'ssid': setup_server.ap_ssid,
                    'password': setup_server.ap_password
                }
                self._send_json(200, ap_data)
            
            def _handle_status(self):
                """Return current setup status"""
                logger.debug(f"[HTTP] Status check: {setup_server._status} - {setup_server._status_message}")
                status_data = {
                    'status': setup_server._status,
                    'message': setup_server._status_message,
                    'error': setup_server._error_details
                }
                self._send_json(200, status_data)
            
            def _handle_networks(self):
                try:
                    logger.info("[HTTP] Starting WiFi network scan...")
                    
                    # Rescan networks
                    logger.debug("[HTTP] Running nmcli wifi rescan...")
                    subprocess.run(['sudo', 'nmcli', 'device', 'wifi', 'rescan'], timeout=5, capture_output=True)
                    asyncio.run(asyncio.sleep(2))
                    
                    # Get network list
                    logger.debug("[HTTP] Fetching WiFi network list...")
                    result = subprocess.run(
                        ['nmcli', '-t', '-f', 'SSID,SECURITY', 'device', 'wifi', 'list'],
                        capture_output=True, text=True, timeout=10
                    )
                    
                    networks = []
                    seen = set()
                    
                    for line in result.stdout.strip().split('\n'):
                        if line.strip():
                            parts = line.split(':', 1)
                            ssid = parts[0].strip()
                            encrypted = bool(parts[1].strip()) if len(parts) > 1 else False
                            
                            if ssid and ssid not in seen:
                                seen.add(ssid)
                                networks.append({'ssid': ssid, 'encrypted': encrypted})
                    
                    logger.info(f"[HTTP] Found {len(networks)} WiFi networks")
                    logger.debug(f"[HTTP] Networks: {[n['ssid'] for n in networks[:5]]}...")
                    self._send_json(200, {'networks': networks})
                    
                except Exception as e:
                    logger.error(f"[HTTP] Error scanning networks: {e}")
                    self._send_json(500, {'networks': [], 'error': str(e)})
            
            def _handle_configure(self):
                try:
                    logger.info("[HTTP] Processing configuration submission...")
                    
                    # Reset status to clear any old error messages
                    setup_server._status = "waiting"
                    setup_server._status_message = "Processing configuration..."
                    setup_server._error_details = None
                    
                    content_length = int(self.headers['Content-Length'])
                    logger.debug(f"[HTTP] Receiving {content_length} bytes of configuration data")
                    post_data = self.rfile.read(content_length)
                    data = json.loads(post_data.decode('utf-8'))
                    
                    ssid = data.get('ssid', '').strip()
                    password = data.get('password', '').strip()
                    pairing_code = data.get('pairing_code', '').strip()
                    
                    logger.info(f"[HTTP] Configuration received:")
                    logger.info(f"[HTTP]   SSID: {ssid}")
                    logger.info(f"[HTTP]   Password: {'*' * len(password) if password else '(empty)'}")
                    logger.info(f"[HTTP]   Pairing Code: {pairing_code}")
                    
                    # Validate
                    if not ssid:
                        logger.warning("[HTTP] Validation failed: SSID is required")
                        self._send_json(400, {'success': False, 'error': 'SSID is required'})
                        return
                    
                    if not pairing_code or len(pairing_code) != 4 or not pairing_code.isdigit():
                        logger.warning(f"[HTTP] Validation failed: Invalid pairing code '{pairing_code}'")
                        self._send_json(400, {'success': False, 'error': 'Valid 4-digit pairing code is required'})
                        return
                    
                    # Call the callback
                    if setup_server._config_callback:
                        logger.info("[HTTP] Calling configuration callback...")
                        asyncio.run(setup_server._config_callback(ssid, password, pairing_code))
                        logger.info("[HTTP] Configuration accepted, sending success response")
                        self._send_json(200, {'success': True})
                    else:
                        logger.error("[HTTP] Configuration callback not set!")
                        self._send_json(500, {'success': False, 'error': 'Server not properly configured'})
                    
                except json.JSONDecodeError as e:
                    logger.error(f"[HTTP] JSON decode error: {e}")
                    self._send_json(400, {'success': False, 'error': 'Invalid JSON data'})
                except Exception as e:
                    logger.error(f"[HTTP] Error handling configuration: {e}", exc_info=True)
                    self._send_json(500, {'success': False, 'error': str(e)})
        
        # Start server in thread
        logger.info(f"[HTTP] Starting HTTP server on 0.0.0.0:{self.port}")
        logger.info(f"[HTTP] AP SSID: {self.ap_ssid}, Password: {self.ap_password}")
        logger.info(f"[HTTP] WiFi Interface: {self.wifi_interface}")
        
        try:
            self._server = TCPServer(('', self.port), SetupHandler)
            self._server_thread = Thread(target=self._server.serve_forever, daemon=True)
            self._server_thread.start()
            logger.info(f"[HTTP] ✓ HTTP server listening on port {self.port}")
            logger.info(f"[HTTP] Access the setup page at: http://192.168.4.1:{self.port}")
        except OSError as e:
            if e.errno == 98:  # Address already in use
                logger.error(f"[HTTP] ✗ Port {self.port} is already in use!")
                logger.error(f"[HTTP] Try: sudo lsof -ti :{self.port} | xargs kill")
                raise
            else:
                logger.error(f"[HTTP] ✗ Failed to start HTTP server: {e}")
                raise
    
    def set_status(self, status: str, message: str, error: str = None):
        """
        Update the status shown to users on the web interface.
        
        Args:
            status: Status code (waiting, connecting, authenticating, success, error)
            message: Human-readable status message
            error: Optional error details
        """
        self._status = status
        self._status_message = message
        self._error_details = error
        
        if error:
            logger.warning(f"[HTTP] Status update: {status} - {message} (Error: {error})")
        else:
            logger.info(f"[HTTP] Status update: {status} - {message}")
    
    async def stop(self):
        """Stop the HTTP server"""
        if self._server:
            logger.info("[HTTP] Stopping HTTP server...")
            try:
                self._server.shutdown()
                self._server.server_close()
                self._server = None
                self._server_thread = None
                logger.info("[HTTP] ✓ HTTP server stopped successfully")
            except Exception as e:
                logger.error(f"[HTTP] Error stopping server: {e}")
        else:
            logger.debug("[HTTP] Server already stopped")

