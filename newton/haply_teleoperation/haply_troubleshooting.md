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
**Current Status**: 3 Haply devices found:
- ✅ 1x `HAPLY_HANDLE_DB0B` (OK status) - **KEEP THIS ONE**
- ❌ 1x `Haply inverse3` (Error status) - **UNINSTALL**
- ❌ 1x `Haply inverse3` (Unknown status) - **UNINSTALL**

**Goal**: Remove the 2 broken "Haply inverse3" devices, keep the working Bluetooth one.

### Step-by-Step Uninstall Process:

#### Step 1: Open Device Manager
1. Press `Windows + R`
2. Type `devmgmt.msc` 
3. Press Enter

#### Step 2: Locate the Problem Devices
**You should find:**
- **1x "Haply inverse3"** under **"Other devices"** ✅ (You found this one!)
- Possibly **1 more "Haply inverse3"** under:
  - **"Universal Serial Bus controllers"**
  - **"Human Interface Devices"**  
  - **"Unknown devices"**
- **DO NOT** touch `HAPLY_HANDLE_DB0B` (this one works!)

**If you only see 1 device**: That's progress! Uninstall the one you found, then check other categories.

#### Step 3: Uninstall the Device in "Other devices"
1. Right-click the **"Haply inverse3"** entry under "Other devices"
2. Select **"Uninstall device"**
3. ✅ **IMPORTANT**: Check **"Delete the driver software for this device"**
4. Click **"Uninstall"**

#### Step 4: Look for Additional Devices
1. **Expand these sections** in Device Manager:
   - "Universal Serial Bus controllers"
   - "Human Interface Devices"
   - Any section with warning icons
2. Look for another **"Haply inverse3"** entry
3. If found, repeat the uninstall process from Step 3

#### Step 5: Verify Cleanup
1. Press **F5** to refresh Device Manager
2. The 2 "Haply inverse3" entries should be **gone**
3. `HAPLY_HANDLE_DB0B` should still be there with **OK status**

#### Step 6: Reconnect Device
1. **Unplug USB cable** from computer
2. **Power off** Haply device (unplug power adapter)  
3. **Wait 30 seconds**
4. **Plug power back in** (wait for LED to turn on)
5. **Plug USB cable back in**
6. Windows should automatically install fresh drivers

#### Step 7: Test Connection
Run this command to verify only 1 working device remains:
```powershell
Get-PnpDevice | Where-Object {$_.FriendlyName -like "*Haply*"}
```
You should see only devices with "OK" status.

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
