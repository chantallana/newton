# Haply Inverse3 Device Troubleshooting Guide

## Power Issues

### 1. Check Power Connection
- Ensure the power adapter is properly connected to the device
- Verify the power cable is securely plugged into a working outlet
- Check if the power LED indicator is lit (usually on the base unit)

### 2. Power Adapter Issues
- Try a different power outlet
- Check if the power adapter LED is on (if it has one)
- Verify the power adapter voltage matches the device requirements
- Try unplugging and reconnecting the power adapter

### 3. USB Connection
- Ensure USB cable is properly connected to both device and computer
- Try a different USB port on your computer
- Use a different USB cable if available
- Check if the device appears in Windows Device Manager

## Device Status Checks

### 4. Check Device Manager
1. Open Windows Device Manager (Right-click Start → Device Manager)
2. Look for "Haply" or "Inverse3" under:
   - Human Interface Devices
   - Universal Serial Bus controllers
   - Other devices (if driver issues)

### 5. Haply Software Status
- Ensure Haply Inverse Service is running
- Check if the Haply Control Panel shows the device
- Verify the device appears in Haply Studio or Haply Console

## Reset Procedures

### 6. Device Reset
- Power off the device completely
- Disconnect USB cable
- Wait 30 seconds
- Reconnect power first, then USB
- Wait for device to initialize (may take 1-2 minutes)

### 7. Software Reset
- Close all Haply applications
- Restart the Haply Inverse Service
- Restart your computer if necessary

## Common Issues

### 8. Driver Issues
- Reinstall Haply drivers from the official website
- Ensure Windows has the latest updates
- Check for Windows driver conflicts

### 9. Firmware Issues
- Check if device firmware needs updating
- Use Haply Studio to check firmware version
- Contact Haply support for firmware recovery if needed

## Current Issue: Duplicate Device Instances

### 10. Multiple Device Instances Detected
**Status**: Two Haply inverse3 instances found:
- One with "Error" status
- One with "Unknown" status

**Solution Steps**:

### Method 1: Device Manager (GUI - Recommended)
1. **Open Device Manager**:
   - Press `Windows + R` → type `devmgmt.msc` → Enter
   - OR Right-click Start button → Device Manager

2. **Find Haply devices**:
   - Look under these sections:
     - "Other devices" (devices with error/unknown status)
     - "Human Interface Devices"
     - "Universal Serial Bus controllers"
   - You should see 2x "Haply inverse3" entries

3. **Uninstall each device**:
   - Right-click first "Haply inverse3" → "Uninstall device"
   - ✅ Check "Delete the driver software for this device"
   - Click "Uninstall"
   - Repeat for the second "Haply inverse3" entry

### Method 2: PowerShell (Advanced)
```powershell
# Run PowerShell as Administrator
Get-PnpDevice | Where-Object {$_.FriendlyName -like "*Haply*"} | Remove-PnpDevice -Force
```
   
2. **Clean USB device history**:
   ```powershell
   # Run as Administrator
   pnputil /delete-driver oem*.inf /uninstall
   ```

3. **Physical reset procedure**:
   - Unplug USB cable from computer
   - Power off Haply device (unplug power adapter)
   - Wait 60 seconds
   - Plug power back into device
   - Wait for LED to stabilize
   - Plug USB cable back into computer

4. **Reinstall drivers**:
   - Download latest Haply drivers
   - Install with administrator privileges

## Contact Information
- Haply Support: support@haply.co
- Documentation: https://docs.haply.co
