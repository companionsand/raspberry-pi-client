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
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 20px;
            padding: 40px;
            max-width: 500px;
            width: 100%;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 { color: #333; margin-bottom: 10px; font-size: 28px; }
        .subtitle { color: #666; margin-bottom: 30px; font-size: 14px; }
        .form-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 8px; color: #333; font-weight: 500; }
        select, input {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
        }
        select:focus, input:focus {
            outline: none;
            border-color: #667eea;
        }
        button {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
        }
        button:disabled { opacity: 0.6; cursor: not-allowed; }
        .message { padding: 12px; border-radius: 8px; margin-bottom: 20px; white-space: pre-line; }
        .success { background: #d4edda; color: #155724; }
        .error { background: #f8d7da; color: #721c24; }
        .info { background: #d1ecf1; color: #0c5460; }
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
                        submitBtn.textContent = 'Try Again';
                    }
                })
                .catch(e => console.error('Status poll failed:', e));
        }
        
        form.onsubmit = async (e) => {
            e.preventDefault();
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
    
    def __init__(self, port: int = 8080, wifi_interface: str = "wlan0"):
        self.port = port
        self.wifi_interface = wifi_interface
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
                logger.debug(f"{self.address_string()} - {format%args}")
            
            def do_GET(self):
                if self.path == '/' or self.path == '/setup.html':
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    self.wfile.write(HTML_TEMPLATE.encode())
                elif self.path == '/networks':
                    self._handle_networks()
                elif self.path == '/status':
                    self._handle_status()
                else:
                    self.send_error(404)
            
            def do_POST(self):
                if self.path == '/configure':
                    self._handle_configure()
                else:
                    self.send_error(404)
            
            def _send_json(self, status_code, data):
                self.send_response(status_code)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())
            
            def _handle_status(self):
                """Return current setup status"""
                status_data = {
                    'status': setup_server._status,
                    'message': setup_server._status_message,
                    'error': setup_server._error_details
                }
                self._send_json(200, status_data)
            
            def _handle_networks(self):
                try:
                    # Rescan networks
                    subprocess.run(['sudo', 'nmcli', 'device', 'wifi', 'rescan'], timeout=5, capture_output=True)
                    asyncio.run(asyncio.sleep(2))
                    
                    # Get network list
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
                    
                    self._send_json(200, {'networks': networks})
                    
                except Exception as e:
                    logger.error(f"Error scanning networks: {e}")
                    self._send_json(500, {'networks': [], 'error': str(e)})
            
            def _handle_configure(self):
                try:
                    content_length = int(self.headers['Content-Length'])
                    post_data = self.rfile.read(content_length)
                    data = json.loads(post_data.decode('utf-8'))
                    
                    ssid = data.get('ssid', '').strip()
                    password = data.get('password', '').strip()
                    pairing_code = data.get('pairing_code', '').strip()
                    
                    # Validate
                    if not ssid:
                        self._send_json(400, {'success': False, 'error': 'SSID is required'})
                        return
                    
                    if not pairing_code or len(pairing_code) != 4 or not pairing_code.isdigit():
                        self._send_json(400, {'success': False, 'error': 'Valid 4-digit pairing code is required'})
                        return
                    
                    # Call the callback
                    if setup_server._config_callback:
                        asyncio.run(setup_server._config_callback(ssid, password, pairing_code))
                        self._send_json(200, {'success': True})
                    else:
                        self._send_json(500, {'success': False, 'error': 'Server not properly configured'})
                    
                except Exception as e:
                    logger.error(f"Error handling configuration: {e}")
                    self._send_json(500, {'success': False, 'error': str(e)})
        
        # Start server in thread
        logger.info(f"Starting HTTP server on port {self.port}...")
        self._server = TCPServer(('', self.port), SetupHandler)
        self._server_thread = Thread(target=self._server.serve_forever, daemon=True)
        self._server_thread.start()
        logger.info("HTTP server started")
    
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
        logger.info(f"Status updated: {status} - {message}")
    
    async def stop(self):
        """Stop the HTTP server"""
        if self._server:
            logger.info("Stopping HTTP server...")
            self._server.shutdown()
            self._server.server_close()
            self._server = None
            self._server_thread = None
            logger.info("HTTP server stopped")

