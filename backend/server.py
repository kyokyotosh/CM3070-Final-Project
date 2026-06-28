import asyncio
import json
from websockets.asyncio.server import serve


async def handler(websocket):
    print("Client connected")
    async for message in websocket:
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            continue

        landmarks = data.get("landmarks", [])

        # Placeholder.
        response = {
            "coaching": f"Backend received {len(landmarks)} landmarks."}
        await websocket.send(json.dumps(response))
    print("Client disconnected")


async def main():
    async with serve(handler, "localhost", 8765) as server:
        print("WebSocket server running on ws://localhost:8765")
        await server.serve_forever()

if __name__ == "__main__":
    asyncio.run(main())
