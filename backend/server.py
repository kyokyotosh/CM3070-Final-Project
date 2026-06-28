import asyncio
import json
from websockets.asyncio.server import serve
from angles import analyze_landmarks


async def handler(websocket):
    print("Client connected")
    async for message in websocket:
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            continue

        landmarks = data.get("landmarks", [])
        angles = analyze_landmarks(landmarks)

        if angles is None:
            response = {
                "coaching": "No pose detected. Make sure your whole body is visible."}
        else:
            response = {
                "coaching": (
                    f"Knee L {angles['left_knee']}° / R {angles['right_knee']}°  |  "
                    f"Trunk lean {angles['trunk_lean']}°"
                ),
                "angles": angles,
            }
        await websocket.send(json.dumps(response))
    print("Client disconnected")


async def main():
    async with serve(handler, "localhost", 8765) as server:
        print("WebSocket server running on ws://localhost:8765")
        await server.serve_forever()

if __name__ == "__main__":
    asyncio.run(main())
