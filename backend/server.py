import asyncio
import json
from websockets.asyncio.server import serve
from angles import analyze_landmarks, analyze_lunge
from squat import SquatAnalyzer
from lunge import LungeAnalyzer
from coach import phrase_feedback


# Manual exercise selection. Until the action-recognition model is integrated,
# the exercise is chosen by the client (a message field) rather than inferred.
EXERCISES = {
    "squat": (analyze_landmarks, SquatAnalyzer),
    "lunge": (analyze_lunge, LungeAnalyzer),
}
DEFAULT_EXERCISE = "squat"


def _make(exercise):
    analyze, analyzer_cls = EXERCISES.get(
        exercise, EXERCISES[DEFAULT_EXERCISE])
    return analyze, analyzer_cls()


async def handler(websocket):
    print("Client connected")
    exercise = DEFAULT_EXERCISE
    analyze, analyzer = _make(exercise)
    last_coaching = "Stand side-on and begin."

    async for message in websocket:
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            continue

        # Allow the client to switch exercise; resets per-session state.
        requested = data.get("exercise", exercise)
        if requested != exercise and requested in EXERCISES:
            exercise = requested
            analyze, analyzer = _make(exercise)
            last_coaching = f"Switched to {exercise}. Stand side-on and begin."

        landmarks = data.get("landmarks", [])
        angles = analyze(landmarks)
        status, verdict = analyzer.update(angles)

        if status["feedback"] is not None:
            last_coaching = status["feedback"]
        if verdict is not None:
            last_coaching = await phrase_feedback(verdict)
            with open("eval_log.csv", "a") as f:
                f.write(
                    f"{exercise},{verdict['rep_number']},"
                    f"{verdict.get('depth_ok')},{verdict.get('trunk_ok')},"
                    f"{verdict.get('knee_travel_ok')},"
                    f"{verdict.get('min_knee')},{verdict.get('max_trunk')},"
                    f"{verdict.get('max_knee_travel')},"
                    f"\"{last_coaching}\"\n"
                )

        response = {
            "exercise": exercise,
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
