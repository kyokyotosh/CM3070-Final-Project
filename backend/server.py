import asyncio
import json
import os
import time

from websockets.asyncio.server import serve

from angles import analyze_landmarks, analyze_lunge
from squat import SquatAnalyzer
from lunge import LungeAnalyzer
from coach import phrase_feedback, detect_faults, select_faults
from quality import score as quality_score, is_partial


# Manual exercise selection. Until the action-recognition model is integrated,
# the exercise is chosen by the client (a message field) rather than inferred.
EXERCISES = {
    "squat": (analyze_landmarks, SquatAnalyzer),
    "lunge": (analyze_lunge, LungeAnalyzer),
}
DEFAULT_EXERCISE = "squat"

# Internal verdict metric names mapped to the keys the interface renders.
# Keeping the mapping on this side means the browser never has to know the
# analyser's internal vocabulary.
METRIC_MAP = {
    "min_knee": "min_knee_angle",
    "max_trunk": "max_trunk_angle",
    "max_knee_travel": "knee_travel",
}

LOG_PATH = "eval_log.csv"
LOG_COLUMNS = [
    "client_ts", "server_ts", "exercise", "rep", "faults", "faults_cued",
    "depth_ok", "trunk_ok", "knee_travel_ok", "min_knee", "max_trunk",
    "max_knee_travel", "quality", "partial", "analysis_ms", "llm_ms",
    "server_ms", "end_to_end_ms", "coaching",
]


def _make(exercise):
    analyze, analyzer_cls = EXERCISES.get(
        exercise, EXERCISES[DEFAULT_EXERCISE])
    return analyze, analyzer_cls()


def _metrics(verdict):
    """The measured values shown beside the verdict in the feed."""
    out = {}
    for source, shown in METRIC_MAP.items():
        value = verdict.get(source)
        if value is not None:
            out[shown] = value
    return out


def _status_message(exercise, coaching, analyzer):
    return {
        "type": "status",
        "exercise": exercise,
        "coaching": coaching,
        "reps": analyzer.rep_count,
        "phase": analyzer.phase,
    }


def _csv_field(value):
    if value is None:
        return ""
    text = str(value)
    if "," in text or '"' in text or "\n" in text:
        return '"' + text.replace('"', '""') + '"'
    return text


def _log_rep(row):
    """Append one completed rep to the evaluation log.

    A header is written when the file is created, so the log is readable
    without a separate schema note. Logging happens after the response has
    been sent, so file access never sits inside the measured latency.
    """
    is_new = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a") as f:
        if is_new:
            f.write(",".join(LOG_COLUMNS) + "\n")
        f.write(",".join(_csv_field(row.get(c)) for c in LOG_COLUMNS) + "\n")


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

        received_ms = time.time() * 1000.0
        message_type = data.get("type", "frame")

        # Allow the client to switch exercise; resets per-session state.
        requested = data.get("exercise", exercise)
        if requested != exercise and requested in EXERCISES:
            exercise = requested
            analyze, analyzer = _make(exercise)
            last_coaching = f"Switched to {exercise}. Stand side-on and begin."

        # The client clears the session: the analyser state must be discarded
        # here too, or the rep counter would carry on from its old value.
        if message_type == "reset":
            analyze, analyzer = _make(exercise)
            last_coaching = "Session cleared. Stand side-on and begin."
            await websocket.send(json.dumps(
                _status_message(exercise, last_coaching, analyzer)))
            continue

        # A bare exercise switch with no landmarks attached.
        if message_type == "exercise":
            await websocket.send(json.dumps(
                _status_message(exercise, last_coaching, analyzer)))
            continue

        landmarks = data.get("landmarks", [])
        analysis_start = time.perf_counter()
        angles = analyze(landmarks)
        status, verdict = analyzer.update(angles)
        analysis_ms = (time.perf_counter() - analysis_start) * 1000.0

        if status["feedback"] is not None:
            last_coaching = status["feedback"]

        if verdict is None:
            await websocket.send(json.dumps({
                "type": "status",
                "exercise": exercise,
                "coaching": last_coaching,
                "reps": status["rep_count"],
                "phase": status["phase"],
            }))
            continue

        # A completed rep. The rule layer has already decided the verdict; the
        # language model only phrases it, and the score and fault list are
        # derived here so the interface never makes a judgement of its own.
        llm_start = time.perf_counter()
        last_coaching = await phrase_feedback(verdict)
        llm_ms = (time.perf_counter() - llm_start) * 1000.0

        faults = detect_faults(verdict)
        faults_cued = [cue for cue, _ in select_faults(verdict)]
        quality = quality_score(verdict)
        partial = is_partial(verdict)

        # End-to-end latency is measured from the moment the browser sent the
        # landmark frame that completed the rep, so it includes transmission
        # in both directions as well as analysis and generation. It does not
        # include the sampling interval between the movement itself and that
        # frame being sent, which adds up to one send period.
        client_ts = data.get("client_ts")
        sent_ms = time.time() * 1000.0
        end_to_end_ms = (round(sent_ms - client_ts, 1)
                         if isinstance(client_ts, (int, float)) else None)

        response = {
            "type": "rep",
            "exercise": exercise,
            "rep": verdict["rep_number"],
            "reps": status["rep_count"],
            "phase": status["phase"],
            "faults": faults,
            "faults_cued": faults_cued,
            "metrics": _metrics(verdict),
            "quality": quality,
            "partial": partial,
            "coaching": last_coaching,
            "latency_ms": end_to_end_ms,
            "timing": {
                "analysis_ms": round(analysis_ms, 1),
                "llm_ms": round(llm_ms, 1),
                "server_ms": round(sent_ms - received_ms, 1),
            },
            "client_ts": client_ts,
        }
        await websocket.send(json.dumps(response))

        _log_rep({
            "client_ts": client_ts,
            "server_ts": round(sent_ms, 1),
            "exercise": exercise,
            "rep": verdict["rep_number"],
            "faults": " ".join(faults),
            "faults_cued": " | ".join(faults_cued),
            "depth_ok": verdict.get("depth_ok"),
            "trunk_ok": verdict.get("trunk_ok"),
            "knee_travel_ok": verdict.get("knee_travel_ok"),
            "min_knee": verdict.get("min_knee"),
            "max_trunk": verdict.get("max_trunk"),
            "max_knee_travel": verdict.get("max_knee_travel"),
            "quality": quality,
            "partial": partial,
            "analysis_ms": round(analysis_ms, 1),
            "llm_ms": round(llm_ms, 1),
            "server_ms": round(sent_ms - received_ms, 1),
            "end_to_end_ms": end_to_end_ms,
            "coaching": last_coaching,
        })

    print("Client disconnected")


async def main():
    async with serve(handler, "localhost", 8765) as server:
        print("WebSocket server running on ws://localhost:8765")
        await server.serve_forever()

if __name__ == "__main__":
    asyncio.run(main())
