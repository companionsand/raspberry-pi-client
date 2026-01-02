"""
Setup HTTP Server

Provides web interface for device setup using asyncio HTTP server.

The interface collects:
- WiFi credentials (SSID, password) - for network connectivity when device has no internet
- Pairing code (4 digits) - for linking device to user account when device is unpaired

Note: WiFi setup and device pairing are conceptually separate, but currently
combined in the same interface for user convenience.
"""

import asyncio
import json
import logging
import os
import subprocess
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from socketserver import TCPServer
from threading import Thread
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Path to HTML template file
TEMPLATE_DIR = Path(__file__).parent / "templates"
TEMPLATE_FILE = TEMPLATE_DIR / "setup.html"


class SetupHTTPServer:
    """HTTP server for device setup interface"""
    
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
        self._pairing_only = False  # If True, only collect pairing code (no WiFi setup)
        self._device_ip: Optional[str] = None  # Device IP address (for pairing-only mode)
    
    async def start(self, config_callback: Callable[[str, str, str], bool], pairing_only: bool = False, device_ip: Optional[str] = None):
        """
        Start the HTTP server.
        
        Args:
            config_callback: Callback function to call when user submits configuration.
                           Should accept (ssid, password, pairing_code) and return bool.
            pairing_only: If True, only collect pairing code (hide WiFi fields, no sudo needed)
            device_ip: Device IP address (for pairing-only mode to show correct URL in logs)
        """
        self._config_callback = config_callback
        self._pairing_only = pairing_only
        self._device_ip = device_ip
        if pairing_only:
            self._status_message = "Waiting for pairing code..."
        
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
                    # Generate HTML based on pairing_only mode
                    html = setup_server._generate_html()
                    self.wfile.write(html.encode())
                elif self.path == '/ap-info':
                    logger.debug(f"[HTTP] Serving AP info to {client_ip}")
                    self._handle_ap_info()
                elif self.path == '/networks':
                    if setup_server._pairing_only:
                        # In pairing-only mode, return empty networks list
                        logger.debug(f"[HTTP] Network scan requested but pairing-only mode - returning empty list")
                        self._send_json(200, {'networks': []})
                    else:
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
                    
                    # Validate (skip SSID check in pairing-only mode)
                    if not setup_server._pairing_only and not ssid:
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
            # Show correct IP based on mode
            if self._pairing_only and self._device_ip:
                logger.info(f"[HTTP] Access the setup page at: http://{self._device_ip}:{self.port}")
            else:
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
    
    def _get_wifi_fields_html(self) -> str:
        """Get WiFi form fields HTML."""
        if self._pairing_only:
            return ''
        return '''
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
            '''
    
    def _get_network_scan_script(self) -> str:
        """Get JavaScript for network scanning."""
        if self._pairing_only:
            return ''
        return '''
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
            '''
    
    def _get_form_validation_script(self, button_text: str) -> str:
        """Get JavaScript for form validation."""
        if self._pairing_only:
            return f'''
                if (!form.pairingCode.value || form.pairingCode.value.length !== 4 || !/^[0-9]{{4}}$/.test(form.pairingCode.value)) {{
                    showMessage('Error: Valid 4-digit pairing code is required', 'error');
                    submitBtn.disabled = false;
                    submitBtn.textContent = '{button_text}';
                    return;
                }}
            '''
        return f'''
                if (!form.ssid.value) {{
                    showMessage('Error: WiFi network is required', 'error');
                    submitBtn.disabled = false;
                    submitBtn.textContent = '{button_text}';
                    return;
                }}
            '''
    
    def _get_html_template_vars(self) -> dict:
        """Get template variables based on pairing_only mode."""
        if self._pairing_only:
            return {
                'wifi_fields_html': self._get_wifi_fields_html(),
                'subtitle': "Enter your pairing code to link this device to your account",
                'button_text': "Pair Device",
                'network_scan_script': self._get_network_scan_script(),
                'form_validation': self._get_form_validation_script("Pair Device"),
                'submit_message': 'Pairing code received. Pairing device...'
            }
        else:
            return {
                'wifi_fields_html': self._get_wifi_fields_html(),
                'subtitle': "Configure your WiFi network to get started",
                'button_text': "Connect",
                'network_scan_script': self._get_network_scan_script(),
                'form_validation': self._get_form_validation_script("Connect"),
                'submit_message': 'Configuration received. Connecting to WiFi...'
            }
    
    def _load_template(self) -> str:
        """Load HTML template from file."""
        try:
            with open(TEMPLATE_FILE, 'r', encoding='utf-8') as f:
                content = f.read()
                # Escape CSS braces that aren't template variables
                # Template variables that should NOT be escaped: subtitle, wifi_fields_html, button_text, 
                # button_text_js, network_scan_script, form_validation, submit_message_js
                import re
                template_vars = ['subtitle', 'wifi_fields_html', 'button_text', 'button_text_js', 
                               'network_scan_script', 'form_validation', 'submit_message_js']
                
                # Strategy: escape all braces, then restore template variables
                # After escaping, {var} becomes {{var}}, so we restore {{var}} back to {var}
                escaped = content.replace('{', '{{').replace('}', '}}')
                # Restore template variables: {{var}} -> {var}
                for var in template_vars:
                    # Match {{var}} (4 braces total) and replace with {var} (2 braces)
                    pattern = '{{' + var + '}}'
                    replacement = '{' + var + '}'
                    escaped = escaped.replace(pattern, replacement)
                
                return escaped
        except FileNotFoundError:
            logger.error(f"Template file not found: {TEMPLATE_FILE}")
            raise
        except Exception as e:
            logger.error(f"Error loading template: {e}")
            raise
    
    def _generate_html(self) -> str:
        """Generate HTML template based on pairing_only mode"""
        vars = self._get_html_template_vars()
        template = self._load_template()
        
        # Replace template variables
        # Escape strings used in JavaScript to prevent injection
        button_text_js = vars['button_text'].replace("'", "\\'").replace('\n', '\\n')
        submit_message_js = vars['submit_message'].replace("'", "\\'").replace('\n', '\\n')
        
        return template.format(
            subtitle=vars['subtitle'],
            wifi_fields_html=vars['wifi_fields_html'],
            button_text=vars['button_text'],
            button_text_js=button_text_js,
            network_scan_script=vars['network_scan_script'],
            form_validation=vars['form_validation'],
            submit_message_js=submit_message_js
        )
    
    async def stop(self):
        """Stop the HTTP server"""
        if self._server:
            logger.info("[HTTP] Stopping HTTP server...")
            try:
                # Run shutdown in executor to avoid blocking the event loop
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._server.shutdown)
                self._server.server_close()
                self._server = None
                self._server_thread = None
                logger.info("[HTTP] ✓ HTTP server stopped successfully")
            except Exception as e:
                logger.error(f"[HTTP] Error stopping server: {e}")
        else:
            logger.debug("[HTTP] Server already stopped")

