import asyncio
import json
from websockets.asyncio.server import serve
from angles import analyze_landmarks
from squat import SquatAnalyzer
from coach import phrase_feedback


async def handler(websocket):
    print("Client connected")
    analyzer = SquatAnalyzer()
    last_coaching = "Stand side-on and begin squatting."
    async for message in websocket:
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            continue

        landmarks = data.get("landmarks", [])
        angles = analyze_landmarks(landmarks)
        status, verdict = analyzer.update(angles)

        if status["feedback"] is not None:
            last_coaching = status["feedback"]
        if verdict is not None:
            last_coaching = await phrase_feedback(verdict)
            with open("eval_log.csv", "a") as f:
                f.write(f"{verdict['rep_number']},{verdict['depth_ok']},"
                        f"{verdict['trunk_ok']},{verdict['min_knee']},"
                        f"{verdict['max_trunk']},\"{last_coaching}\"\n")

        response = {
            "coaching": last_coaching,
            "reps": status["rep_count"],
            "phase": status["phase"],
        }
        await websocket.send(json.dumps(response))
    print("Client disconnected")


async def main():
    async with serve(handler, "localhost", 8765) as server:
        print("WebSocket server running on ws://localhost:8765")
        await server.serve_forever()

if __name__ == "__main__":
    asyncio.run(main())
