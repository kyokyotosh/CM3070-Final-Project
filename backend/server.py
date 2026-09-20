import asyncio
import json
import os
import time

from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from angles import analyze_landmarks, analyze_lunge
from squat import SquatAnalyzer
from lunge import LungeAnalyzer
from coach import phrase_feedback, detect_faults, select_faults, warm_up
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
    "max_knee_travel", "quality", "partial", "analysis_ms", "verdict_ms",
    "queue_ms", "llm_ms", "cue_ms", "coaching",
]

# The client streams at 15 Hz for the action recogniser. Form analysis runs
# on every third frame, preserving the 200 ms interval its thresholds were
# calibrated against and keeping latency results comparable across versions.
ANALYSIS_STRIDE = 3


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


def _elapsed_since(client_ts, now_ms):
    """Milliseconds from the browser sending a frame to this moment, or None
    when the client did not stamp the frame."""
    if not isinstance(client_ts, (int, float)):
        return None
    return round(now_ms - client_ts, 1)


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
    without a separate schema note. Logging happens in the cue worker, after
    both messages have been queued for sending, so file access never sits
    inside a measured latency.
    """
    is_new = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a") as f:
        if is_new:
            f.write(",".join(LOG_COLUMNS) + "\n")
        f.write(",".join(_csv_field(row.get(c)) for c in LOG_COLUMNS) + "\n")


async def _writer(websocket, outbox):
    """Single owner of the socket.

    Verdicts and cues are produced by different coroutines, so they are queued
    here rather than sent from wherever they were created. One writer also
    keeps the order of messages predictable.
    """
    while True:
        message = await outbox.get()
        try:
            await websocket.send(json.dumps(message))
        except ConnectionClosed:
            return


async def _cue_worker(cue_queue, outbox, session):
    """Phrase verdicts into coaching cues, one at a time, off the read loop.

    Generation is the slowest stage in the pipeline by three orders of
    magnitude. Running it here means an in-flight generation no longer delays
    the reading of incoming frames, so the next rep is still detected and its
    verdict still delivered while the previous sentence is being written.

    Jobs are handled one after another rather than concurrently: the model
    serves one request at a time in any case, and a queue keeps cues in rep
    order. The wait is measured and reported as queue_ms rather than hidden.
    """
    while True:
        job = await cue_queue.get()

        # A cleared or switched session discards work queued before it, so a
        # stale sentence cannot arrive after the panel has been reset.
        if job["epoch"] != session["epoch"]:
            continue

        queue_ms = round((time.perf_counter() - job["queued_at"]) * 1000.0, 1)

        llm_start = time.perf_counter()
        text = await phrase_feedback(job["verdict"])
        llm_ms = round((time.perf_counter() - llm_start) * 1000.0, 1)

        if job["epoch"] != session["epoch"]:
            continue

        session["coaching"] = text
        sent_ms = time.time() * 1000.0
        cue_ms = _elapsed_since(job["client_ts"], sent_ms)

        await outbox.put({
            "type": "cue",
            "exercise": job["exercise"],
            "rep": job["verdict"]["rep_number"],
            "coaching": text,
            "latency_ms": cue_ms,
            "timing": {"queue_ms": queue_ms, "llm_ms": llm_ms},
        })

        row = dict(job["row"])
        row.update({
            "server_ts": round(sent_ms, 1),
            "queue_ms": queue_ms,
            "llm_ms": llm_ms,
            "cue_ms": cue_ms,
            "coaching": text,
        })
        _log_rep(row)


async def handler(websocket):
    print("Client connected")
    exercise = DEFAULT_EXERCISE
    analyze, analyzer = _make(exercise)

    # Shared with the cue worker: the latest coaching line, and the epoch that
    # invalidates work queued before a reset or an exercise switch.
    session = {"coaching": "Stand side-on and begin.", "epoch": 0}

    outbox = asyncio.Queue()
    cue_queue = asyncio.Queue()
    writer = asyncio.create_task(_writer(websocket, outbox))
    cue_worker = asyncio.create_task(_cue_worker(cue_queue, outbox, session))
    frame_index = 0

    try:
        async for message in websocket:
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                continue

            message_type = data.get("type", "frame")

            # Allow the client to switch exercise; resets per-session state.
            requested = data.get("exercise", exercise)
            if requested != exercise and requested in EXERCISES:
                exercise = requested
                analyze, analyzer = _make(exercise)
                session["epoch"] += 1
                session["coaching"] = (
                    f"Switched to {exercise}. Stand side-on and begin.")

            # The client clears the session: the analyser state must be
            # discarded here too, or the rep counter would carry on from its
            # old value.
            if message_type == "reset":
                analyze, analyzer = _make(exercise)
                session["epoch"] += 1
                session["coaching"] = "Session cleared. Stand side-on and begin."

            if message_type in ("reset", "exercise"):
                await outbox.put({
                    "type": "status",
                    "exercise": exercise,
                    "coaching": session["coaching"],
                    "reps": analyzer.rep_count,
                    "phase": analyzer.phase,
                })
                continue

            landmarks = data.get("landmarks", [])
            client_ts = data.get("client_ts")

            frame_index += 1
            if frame_index % ANALYSIS_STRIDE != 0:
                continue

            analysis_start = time.perf_counter()
            angles = analyze(landmarks)
            status, verdict = analyzer.update(angles)
            analysis_ms = round(
                (time.perf_counter() - analysis_start) * 1000.0, 2)

            if status["feedback"] is not None:
                session["coaching"] = status["feedback"]

            if verdict is None:
                await outbox.put({
                    "type": "status",
                    "exercise": exercise,
                    "coaching": session["coaching"],
                    "reps": status["rep_count"],
                    "phase": status["phase"],
                })
                continue

            # A completed rep. The rule layer has already decided everything
            # the user needs to see, so the verdict goes out now and the
            # language model's sentence follows as a separate message. Nothing
            # the interface renders waits on generation except the wording.
            faults = detect_faults(verdict)
            faults_cued = [cue for cue, _ in select_faults(verdict)]
            quality = quality_score(verdict)
            partial = is_partial(verdict)

            verdict_sent_ms = time.time() * 1000.0
            await outbox.put({
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
                "latency_ms": _elapsed_since(client_ts, verdict_sent_ms),
                "timing": {"analysis_ms": analysis_ms},
                "client_ts": client_ts,
            })

            await cue_queue.put({
                "verdict": verdict,
                "exercise": exercise,
                "client_ts": client_ts,
                "epoch": session["epoch"],
                "queued_at": time.perf_counter(),
                "row": {
                    "client_ts": client_ts,
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
                    "analysis_ms": analysis_ms,
                    "verdict_ms": _elapsed_since(client_ts, verdict_sent_ms),
                },
            })
    finally:
        cue_worker.cancel()
        writer.cancel()
        await asyncio.gather(cue_worker, writer, return_exceptions=True)
        print("Client disconnected")


async def main():
    # Load the model before the first user rep, so its one-off initialisation
    # is not charged to that rep's latency.
    await warm_up()
    async with serve(handler, "localhost", 8765) as server:
        print("WebSocket server running on ws://localhost:8765")
        await server.serve_forever()

if __name__ == "__main__":
    asyncio.run(main())
