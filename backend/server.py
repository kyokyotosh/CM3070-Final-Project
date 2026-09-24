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

try:
    from recogniser import (DEFAULT_MODEL_PATH, ExerciseGate, Recogniser,
                            EXERCISE_CLASSES)
    RECOGNITION_AVAILABLE = True
except Exception as exc:                      # torch or weights missing
    print(f"Exercise recognition unavailable, manual selection only: {exc}")
    RECOGNITION_AVAILABLE = False


EXERCISES = {
    "squat": (analyze_landmarks, SquatAnalyzer),
    "lunge": (analyze_lunge, LungeAnalyzer),
}
DEFAULT_EXERCISE = "squat"

# The client streams at 15 Hz for the recogniser, which was trained at that
# rate. Form analysis runs on every third frame, preserving the 200 ms
# interval its thresholds were calibrated against and keeping latency results
# comparable with the versions measured before recognition was added.
ANALYSIS_STRIDE = 3

# Classify every fifth frame, about three times a second. Windows are two
# seconds long and overlap heavily, so a higher rate would spend GPU time to
# re-decide almost the same question. This matches the stride the model was
# trained with.
RECOGNITION_STRIDE = 5

METRIC_MAP = {
    "min_knee": "min_knee_angle",
    "max_trunk": "max_trunk_angle",
    "max_knee_travel": "knee_travel",
}

LOG_PATH = "eval_log.csv"
LOG_COLUMNS = [
    "client_ts", "server_ts", "mode", "exercise", "detected", "detect_conf",
    "rep", "faults", "faults_cued", "depth_ok", "trunk_ok", "knee_travel_ok",
    "min_knee", "max_trunk", "max_knee_travel", "quality", "partial",
    "analysis_ms", "verdict_ms", "queue_ms", "llm_ms", "cue_ms", "coaching",
]


def _make_analyzers():
    """One analyser per exercise, held for the life of the session.

    Switching exercise selects a different analyser rather than building one,
    so a switch never discards a rep count. The first version rebuilt on every
    switch, which meant one spurious detection wiped the set irreversibly: the
    cost of a false switch has to stay proportional to the mistake.
    """
    return {name: (fn, cls()) for name, (fn, cls) in EXERCISES.items()}


def _total_reps(analyzers):
    return sum(analyzer.rep_count for _, analyzer in analyzers.values())


def _metrics(verdict):
    out = {}
    for source, shown in METRIC_MAP.items():
        value = verdict.get(source)
        if value is not None:
            out[shown] = value
    return out


def _elapsed_since(client_ts, now_ms):
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
    is_new = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a") as f:
        if is_new:
            f.write(",".join(LOG_COLUMNS) + "\n")
        f.write(",".join(_csv_field(row.get(c)) for c in LOG_COLUMNS) + "\n")


async def _writer(websocket, outbox):
    """Single owner of the socket, so tasks queue messages instead of sending."""
    while True:
        message = await outbox.get()
        try:
            await websocket.send(json.dumps(message))
        except ConnectionClosed:
            return


async def _recognition_worker(inbox, results, recogniser):
    """Classify windows away from the read loop.

    Inference takes tens of milliseconds. Running it inline would stall the
    reading of landmark frames, which is the same mistake the language model
    made before generation was moved off this path. Results are posted to a
    queue and applied by the read loop, which owns the analyser.
    """
    while True:
        window, epoch = await inbox.get()
        try:
            label, confidence, _ = await asyncio.to_thread(
                recogniser.classify, window)
        except Exception as exc:
            print(f"recognition failed: {exc}")
            continue
        await results.put((label, confidence, epoch))


async def _cue_worker(cue_queue, outbox, session):
    """Phrase verdicts into coaching cues, one at a time, off the read loop."""
    while True:
        job = await cue_queue.get()
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
        row.update({"server_ts": round(sent_ms, 1), "queue_ms": queue_ms,
                    "llm_ms": llm_ms, "cue_ms": cue_ms, "coaching": text})
        _log_rep(row)


async def handler(websocket):
    print("Client connected")

    mode = DEFAULT_EXERCISE          # "squat", "lunge" or "auto"
    exercise = DEFAULT_EXERCISE      # the analyser actually running
    analyzers = _make_analyzers()
    analyze, analyzer = analyzers[exercise]

    session = {"coaching": "Stand side-on and begin.", "epoch": 0}

    outbox = asyncio.Queue()
    cue_queue = asyncio.Queue()
    tasks = [asyncio.create_task(_writer(websocket, outbox)),
             asyncio.create_task(_cue_worker(cue_queue, outbox, session))]

    recogniser = gate = None
    recognition_inbox = recognition_results = None
    if RECOGNITION_AVAILABLE:
        try:
            recogniser = Recogniser(DEFAULT_MODEL_PATH)
            gate = ExerciseGate(default=DEFAULT_EXERCISE)
            recognition_inbox = asyncio.Queue(maxsize=1)
            recognition_results = asyncio.Queue()
            tasks.append(asyncio.create_task(_recognition_worker(
                recognition_inbox, recognition_results, recogniser)))
            print(f"recognition ready on {recogniser.device} "
                  f"({recogniser.window_size} frames, {recogniser.mode} "
                  f"normalisation)")
        except Exception as exc:
            print(f"recognition disabled: {exc}")
            recogniser = gate = None

    frame_index = 0
    detected, detect_conf, suppressed = None, 0.0, False

    try:
        async for message in websocket:
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                continue

            message_type = data.get("type", "frame")
            requested = data.get("exercise", mode)

            # Mode changes: an explicit exercise, or "auto" to let the
            # recogniser choose. Either way the analyser is rebuilt, so the
            # rep count restarts, and queued cues are invalidated.
            if requested != mode and (requested in EXERCISES
                                      or requested == "auto"):
                mode = requested
                if mode in EXERCISES:
                    exercise = mode
                # An explicit choice by the person starts a fresh session.
                analyzers = _make_analyzers()
                analyze, analyzer = analyzers[exercise]
                session["epoch"] += 1
                if recogniser:
                    recogniser.reset()
                if gate:
                    gate.reset(default=exercise)
                detected, detect_conf, suppressed = None, 0.0, False
                session["coaching"] = (
                    "Detecting the exercise. Stand side-on and begin."
                    if mode == "auto"
                    else f"Switched to {exercise}. Stand side-on and begin.")

            if message_type == "reset":
                analyzers = _make_analyzers()
                analyze, analyzer = analyzers[exercise]
                session["epoch"] += 1
                if recogniser:
                    recogniser.reset()
                if gate:
                    gate.reset(default=exercise)
                detected, detect_conf, suppressed = None, 0.0, False
                session["coaching"] = "Session cleared. Stand side-on and begin."

            if message_type in ("reset", "exercise"):
                await outbox.put({
                    "type": "status", "mode": mode, "exercise": exercise,
                    "detected": detected, "detect_confidence": detect_conf,
                    "suppressed": suppressed,
                    "coaching": session["coaching"],
                    "reps": analyzer.rep_count,
                    "reps_total": _total_reps(analyzers),
                    "phase": analyzer.phase,
                })
                continue

            landmarks = data.get("landmarks", [])
            client_ts = data.get("client_ts")
            frame_index += 1

            # Recognition sees every frame; it was trained at the full rate.
            if recogniser is not None:
                recogniser.push(landmarks)
                if (frame_index % RECOGNITION_STRIDE == 0
                        and recogniser.ready
                        and not recognition_inbox.full()):
                    window = recogniser.snapshot()
                    if window is not None:
                        recognition_inbox.put_nowait(
                            (window, session["epoch"]))

            # Apply whatever the worker has finished, while the analyser is
            # in a known phase.
            while recognition_results is not None and not recognition_results.empty():
                label, confidence, epoch = recognition_results.get_nowait()
                if epoch != session["epoch"]:
                    continue
                decision = gate.update(label, confidence, analyzer.phase)
                detected = decision["detected"]
                detect_conf = round(decision["confidence"], 3)
                # Only automatic mode acts on the detection. In manual mode it
                # is reported but never overrides the person's choice, which
                # also keeps the two modes separately measurable.
                if mode == "auto":
                    suppressed = decision["suppressed"]
                    if decision["switched"]:
                        exercise = decision["exercise"]
                        # Select the other analyser, keeping its count. The
                        # epoch is deliberately not bumped: a cue already
                        # queued describes a repetition that did happen, and
                        # discarding it would lose real feedback.
                        analyze, analyzer = analyzers[exercise]
                        session["coaching"] = (
                            f"{exercise.capitalize()} detected, "
                            f"{analyzer.rep_count} so far.")
                else:
                    suppressed = False

            # Form analysis on every third frame only.
            if frame_index % ANALYSIS_STRIDE != 0:
                continue

            analysis_start = time.perf_counter()
            angles = analyze(landmarks)
            status, verdict = analyzer.update(angles)
            analysis_ms = round(
                (time.perf_counter() - analysis_start) * 1000.0, 2)

            if status["feedback"] is not None:
                session["coaching"] = status["feedback"]

            # A repetition detected while the recogniser says the person is
            # not exercising is discarded rather than coached. This is the
            # third class doing its job: sitting or stretching flexes the
            # knees enough to complete the state machine.
            if verdict is not None and suppressed:
                verdict = None
                session["coaching"] = "Not exercising, so that was not counted."

            if verdict is None:
                await outbox.put({
                    "type": "status", "mode": mode, "exercise": exercise,
                    "detected": detected, "detect_confidence": detect_conf,
                    "suppressed": suppressed,
                    "coaching": session["coaching"],
                    "reps": status["rep_count"],
                    "reps_total": _total_reps(analyzers),
                    "phase": status["phase"],
                })
                continue

            faults = detect_faults(verdict)
            faults_cued = [cue for cue, _ in select_faults(verdict)]
            quality = quality_score(verdict)
            partial = is_partial(verdict)

            verdict_sent_ms = time.time() * 1000.0
            await outbox.put({
                "type": "rep", "mode": mode, "exercise": exercise,
                "detected": detected, "detect_confidence": detect_conf,
                "rep": verdict["rep_number"], "reps": status["rep_count"],
                "reps_total": _total_reps(analyzers),
                "phase": status["phase"], "faults": faults,
                "faults_cued": faults_cued, "metrics": _metrics(verdict),
                "quality": quality, "partial": partial,
                "latency_ms": _elapsed_since(client_ts, verdict_sent_ms),
                "timing": {"analysis_ms": analysis_ms},
                "client_ts": client_ts,
            })

            await cue_queue.put({
                "verdict": verdict, "exercise": exercise,
                "client_ts": client_ts, "epoch": session["epoch"],
                "queued_at": time.perf_counter(),
                "row": {
                    "client_ts": client_ts, "mode": mode, "exercise": exercise,
                    "detected": detected, "detect_conf": detect_conf,
                    "rep": verdict["rep_number"], "faults": " ".join(faults),
                    "faults_cued": " | ".join(faults_cued),
                    "depth_ok": verdict.get("depth_ok"),
                    "trunk_ok": verdict.get("trunk_ok"),
                    "knee_travel_ok": verdict.get("knee_travel_ok"),
                    "min_knee": verdict.get("min_knee"),
                    "max_trunk": verdict.get("max_trunk"),
                    "max_knee_travel": verdict.get("max_knee_travel"),
                    "quality": quality, "partial": partial,
                    "analysis_ms": analysis_ms,
                    "verdict_ms": _elapsed_since(client_ts, verdict_sent_ms),
                },
            })
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        print("Client disconnected")


async def main():
    await warm_up()
    async with serve(handler, "localhost", 8765) as server:
        print("WebSocket server running on ws://localhost:8765")
        await server.serve_forever()

if __name__ == "__main__":
    asyncio.run(main())
