#!/usr/bin/env python3
"""Test Verse Grip button detection"""

import asyncio
import json
try:
    import orjson
except ImportError:
    import json as orjson
try:
    import websockets
except ImportError:
    print("ERROR: websockets module not found. Install with: uv add websockets")
    exit(1)

async def test_buttons():
    uri = "ws://localhost:10001"
    print("Testing Verse Grip button detection...")
    print("Press and hold different buttons on your Verse Grip device.")
    print("Press Ctrl+C to exit.")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to Haply service!")
            
            # Request device updates
            await websocket.send('{"request": "deviceUpdate"}')
            
            while True:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                    data = orjson.loads(response) if hasattr(orjson, 'loads') else json.loads(response)
                    
                    # Check for Verse Grip devices
                    verse_grip_devices = data.get("wireless_verse_grip", [])
                    if verse_grip_devices:
                        verse_grip_data = verse_grip_devices[0]
                        buttons = verse_grip_data.get("state", {}).get("buttons", {})
                        
                        if buttons:
                            # Check if any buttons are pressed
                            pressed_buttons = [key for key, value in buttons.items() if value]
                            if pressed_buttons:
                                print(f"🔘 Buttons pressed: {pressed_buttons}")
                                print(f"    Raw buttons dict: {buttons}")
                                
                except asyncio.TimeoutError:
                    continue
                except KeyboardInterrupt:
                    print("\nExiting...")
                    break
                    
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("Make sure the Haply Inverse Service is running.")

if __name__ == "__main__":
    try:
        asyncio.run(test_buttons())
    except KeyboardInterrupt:
        print("\nTest completed.")
