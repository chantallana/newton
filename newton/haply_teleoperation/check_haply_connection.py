#!/usr/bin/env python3
"""
Simple Haply device connection checker
"""
import asyncio
import websockets
import orjson
import sys

async def check_haply_connection():
    uri = 'ws://localhost:10001'
    
    try:
        print("Attempting to connect to Haply service...")
        async with websockets.connect(uri, open_timeout=10) as ws:
            print("✅ Connected to Haply service!")
            
            # Try to receive one message
            try:
                response = await asyncio.wait_for(ws.recv(), timeout=5)
                data = orjson.loads(response)
                
                inverse3_devices = data.get("inverse3", [])
                verse_grip_devices = data.get("wireless_verse_grip", [])
                
                if inverse3_devices:
                    device = inverse3_devices[0]
                    device_id = device.get("device_id", "Unknown")
                    print(f"✅ Inverse3 device detected! ID: {device_id}")
                    
                    if "config" in device:
                        handedness = device["config"].get("handedness", "Unknown")
                        print(f"   Handedness: {handedness}")
                else:
                    print("❌ No Inverse3 device found")
                
                if verse_grip_devices:
                    device = verse_grip_devices[0]
                    device_id = device.get("device_id", "Unknown")
                    print(f"✅ Verse Grip device detected! ID: {device_id}")
                else:
                    print("❌ No Verse Grip device found")
                    
            except asyncio.TimeoutError:
                print("❌ No data received from Haply service (timeout)")
                
    except websockets.exceptions.ConnectionRefusedError:
        print("❌ Cannot connect to Haply service - is it running?")
    except Exception as e:
        print(f"❌ Connection error: {e}")

if __name__ == "__main__":
    print("Haply Device Connection Checker")
    print("=" * 40)
    asyncio.run(check_haply_connection())
