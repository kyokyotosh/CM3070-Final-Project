import asyncio
import json
from websockets.asyncio.server import serve
from angles import analyze_landmarks
from squat import SquatAnalyzer


async def handler(websocket):
    print("Client connected")
    analyzer = SquatAnalyzer()
    async for message in websocket:
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            continue

        landmarks = data.get("landmarks", [])
        angles = analyze_landmarks(landmarks)
        status = analyzer.update(angles)

        response = {
            "coaching": status["feedback"],
            "reps": status["rep_count"],
            "phase": status["phase"],
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
