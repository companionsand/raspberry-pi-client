# WiFi Setup Flow Improvements

## Problem Statement

Previously, when WiFi setup partially succeeded (WiFi connected but authentication failed), the system would:

- ❌ Keep the WiFi connection saved in NetworkManager
- ❌ Keep the pairing code in memory
- ❌ Create corrupt connection profiles on retry
- ❌ Fail to reconnect due to "key-mgmt missing" errors

## Solution

The flow has been improved to ensure **complete cleanup** on any failure.

## New Flow

### Happy Path (Everything Succeeds)

```
1. User enters WiFi credentials + pairing code
   ↓
2. Device connects to WiFi ✓
   ↓
3. Device authenticates with pairing code ✓
   ↓
4. WiFi setup exits successfully
   ↓
5. Device starts normal operation
```

### Failure Path 1: Wrong WiFi Password

```
1. User enters WiFi credentials + pairing code
   ↓
2. Device tries to connect to WiFi
   ↓
3. Connection fails ✗
   ↓
4. network_connector.py DELETES the failed connection profile
   ↓
5. Manager shows error message to user
   ↓
6. Access point restarts automatically
   ↓
7. User can try again with correct password
```

### Failure Path 2: WiFi Succeeds, Authentication Fails

```
1. User enters WiFi credentials + pairing code
   ↓
2. Device connects to WiFi ✓
   ↓
3. Device tries to authenticate with pairing code
   ↓
4. Authentication fails ✗ (wrong code, expired, etc.)
   ↓
5. main.py DELETES the WiFi connection
   ↓
6. main.py CLEARS credentials from memory
   ↓
7. Access point restarts automatically
   ↓
8. User can try again with correct pairing code
```

## Code Changes

### 1. network_connector.py - Auto-Cleanup Failed Connections

**When connection fails:**

```python
# Delete the failed connection to avoid corrupt profiles
logger.debug(f"[NetConnect] Cleaning up failed connection for {ssid}...")
await self._run_sudo_cmd([
    'nmcli', 'connection', 'delete', ssid
], check=False, suppress_output=True)
```

**Benefits:**

- Prevents "key-mgmt missing" errors
- Prevents corrupt connection profiles
- Ensures clean state for retry

### 2. main.py - Cleanup on Authentication Failure

**When authentication fails:**

```python
# Clean up the failed WiFi connection
if wifi_manager._wifi_credentials:
    failed_ssid = wifi_manager._wifi_credentials[0]
    print(f"  Deleting failed connection: {failed_ssid}")
    subprocess.run(['sudo', 'nmcli', 'connection', 'delete', failed_ssid])

# Clear credentials from manager
wifi_manager._wifi_credentials = None
wifi_manager._pairing_code = None
```

**Benefits:**

- Ensures retry starts from scratch
- No leftover credentials from failed attempt
- User must re-enter both WiFi and pairing code

## User Experience

### Before (Problems):

1. User enters wrong pairing code
2. Authentication fails
3. WiFi connection stays saved
4. Retry attempts fail with "key-mgmt missing"
5. User must manually delete connections
6. Confusing error messages

### After (Improved):

1. User enters wrong pairing code
2. Authentication fails
3. **System automatically deletes WiFi connection**
4. **System clears all credentials**
5. Access point restarts cleanly
6. User sees clear message: "Please reconnect to Kin_Setup and try again"
7. User can retry with fresh state

## Fixing Existing Corrupt Connections

If you already have a corrupt connection, fix it manually:

```bash
# List all connections
sudo nmcli connection show

# Delete the corrupt one
sudo nmcli connection delete "YourNetworkName"

# Now you can connect cleanly
sudo nmcli device wifi connect "YourNetworkName" password "your-password" ifname wlan0
```

## Testing Scenarios

### Test 1: Wrong WiFi Password

```
Expected: Connection fails, profile deleted, can retry immediately
```

### Test 2: Correct WiFi, Wrong Pairing Code

```
Expected: WiFi connects, auth fails, WiFi deleted, can retry from scratch
```

### Test 3: Multiple Failed Attempts

```
Expected: Each attempt cleans up completely, no leftover state
```

### Test 4: Network Temporarily Unavailable

```
Expected: Connection times out, profile deleted, can retry when network available
```

## Logging

You'll now see clear cleanup messages:

```
[NetConnect] ✗ Connection failed: Secrets were required
[NetConnect] Cleaning up failed connection for HomeWiFi...
```

```
✗ Authentication or pairing failed
  Cleaning up and restarting setup mode...
  Deleting failed connection: HomeWiFi
  Please reconnect to Kin_Setup and try again
```

## Benefits Summary

✅ **Automatic cleanup** - No manual intervention needed
✅ **Clean retries** - Each attempt starts fresh
✅ **No corrupt profiles** - Prevents "key-mgmt missing" errors
✅ **Clear messaging** - User knows what's happening
✅ **Idempotent** - Safe to retry indefinitely
✅ **Fail-safe** - Handles all error cases gracefully

## Important Note

The setup will only succeed and exit when **BOTH** of these are true:

1. WiFi connection succeeds AND has internet
2. Device authentication succeeds

If either fails, everything is cleaned up and you start fresh.
