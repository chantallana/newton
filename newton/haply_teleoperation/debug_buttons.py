#!/usr/bin/env python3
"""Debug Haply button detection in Newton haptics"""

import asyncio
import threading
import time
import json
try:
    import orjson
except ImportError:
    orjson = json
try:
    import websockets
except ImportError:
    print("ERROR: websockets module not found. Install with: uv add websockets")
    exit(1)

class HaplyButtonDebugger:
    def __init__(self):
        self._clutch_active = False
        self._button_b_active = False
        self._prev_button_c = False
        self._feedback_lock = threading.Lock()
        
    async def haply_debug_loop(self):
        uri = "ws://localhost:10001"
        print("🔍 Haply Button Debugger")
        print("=" * 50)
        print("Controls:")
        print("  Button A (Clutch): Should enable/disable control")
        print("  Button B: Should be detected")  
        print("  Button C: Should be detected")
        print("Press buttons on your Verse Grip to test detection...")
        print()
        
        try:
            async with websockets.connect(uri) as ws:
                print("✅ Connected to Haply service")
                await ws.send('{"request": "deviceUpdate"}')
                
                buttons_printed = False
                prev_clutch = False
                
                while True:
                    try:
                        response = await asyncio.wait_for(ws.recv(), timeout=0.1)
                        data = orjson.loads(response) if hasattr(orjson, 'loads') else json.loads(response)
                        
                        # Extract device data 
                        inverse3_devices = data.get("inverse3", [])
                        verse_grip_devices = data.get("wireless_verse_grip", [])
                        verse_grip_data = verse_grip_devices[0] if verse_grip_devices else {}
                        
                        if not verse_grip_data:
                            continue
                            
                        # Get button states
                        buttons = verse_grip_data.get("state", {}).get("buttons", {})
                        if not buttons:
                            continue
                            
                        # Print raw buttons dict once for debugging
                        if buttons and not buttons_printed:
                            print(f"🔍 Raw buttons dict: {buttons}")
                            buttons_printed = True
                            
                        # Newton's exact button detection logic
                        clutch = bool(
                            buttons.get("1", False)
                            or buttons.get("b1", False) 
                            or buttons.get("a", False)
                            or buttons.get("primary", False)
                        )
                        button_b = bool(
                            buttons.get("2", False)
                            or buttons.get("b2", False)
                            or buttons.get("b", False)
                        )
                        button_c = bool(
                            buttons.get("3", False)
                            or buttons.get("b3", False)
                            or buttons.get("c", False)
                        )
                        
                        # Show button states when they change
                        if clutch != prev_clutch:
                            if clutch:
                                print("🟢 CLUTCH ENGAGED (Button A pressed)")
                                print("   → Control should now be active!")
                            else:
                                print("🔴 CLUTCH RELEASED (Button A released)")
                                print("   → Control should now be disabled")
                                
                        if button_b:
                            print("🔵 Button B pressed")
                            
                        if button_c:
                            print("🟡 Button C pressed")
                            
                        # Update state
                        with self._feedback_lock:
                            self._clutch_active = clutch
                            self._button_b_active = button_b
                            
                        prev_clutch = clutch
                        
                    except asyncio.TimeoutError:
                        continue
                    except KeyboardInterrupt:
                        print("\nExiting...")
                        break
                        
        except Exception as e:
            print(f"❌ Connection failed: {e}")

async def main():
    debugger = HaplyButtonDebugger()
    try:
        await debugger.haply_debug_loop()
    except KeyboardInterrupt:
        print("\nButton test completed.")

if __name__ == "__main__":
    asyncio.run(main())
