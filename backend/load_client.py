"""Synthetic load client for latency and rep-counting measurement.

Drives the running server over the real WebSocket with generated landmark
frames, at a rep cadence set on the command line. This exists because the
interesting latency question cannot be answered by exercising in front of the
camera: the queue behind the language model only builds when repetitions
arrive faster than cues can be generated, which is faster than a person can
squat. It also gives an exact expected repetition count, so the counter can be
checked at speed rather than against a tally kept while moving.

Landmarks are synthesised from the target joint angles rather than replayed
from a recording, so the geometry stage is exercised on the same path as a
real session: the client sends coordinates and the server derives every angle
itself.

Usage, with the server already running:

    python3 load_client.py --reps 20 --period 1000
    python3 load_client.py --reps 20 --period 600
    python3 load_client.py --reps 20 --period 400 --out load_400ms.csv

Defaults produce clean repetitions. Pass --bottom 110 for a depth fault or
--trunk 60 for a trunk fault, to measure a session where every rep carries a
fault and the coaching text is longer.
"""

import argparse
import asyncio
import json
import math
import statistics
import time

URL = "ws://localhost:8765"

# Frame rate of the real client. Landmarks are sent on this interval so the
# measured path matches the deployed configuration.
FRAME_INTERVAL_MS = 200

# Fixed skeleton geometry, in normalised image coordinates. The hip and ankle
# stay put and the knee is moved to produce the requested angle.
HIP_Y, KNEE_Y, ANKLE_Y = 0.45, 0.65, 0.85
SHOULDER_Y = 0.25
CENTRE_X = 0.5
STANDING_KNEE = 170.0


def _landmark(x, y, vis=1.0):
    return {"x": x, "y": y, "z": 0.0, "visibility": vis}


def _knee_offset(angle_deg):
    """Horizontal knee displacement that produces the requested knee angle.

    With the hip directly above the ankle and the knee halfway between them,
    the interior angle at the knee is 2 * atan(half_span / offset), so the
    offset needed for a given angle is half_span / tan(angle / 2).
    """
    half_span = (ANKLE_Y - HIP_Y) / 2.0
    half_angle = math.radians(angle_deg / 2.0)
    if half_angle <= 0:
        return 0.0
    return half_span / math.tan(half_angle)


def _shoulder_offset(trunk_deg):
    """Horizontal shoulder displacement that produces the requested trunk
    lean away from vertical."""
    rise = HIP_Y - SHOULDER_Y
    return rise * math.tan(math.radians(trunk_deg))


def pose(knee_deg, trunk_deg):
    """A 33-landmark side-on pose with the requested knee angle and trunk
    lean. Both legs are given the same geometry, so leg selection is stable."""
    knee_x = CENTRE_X - _knee_offset(knee_deg)
    shoulder_x = CENTRE_X - _shoulder_offset(trunk_deg)

    lm = [_landmark(CENTRE_X, 0.5) for _ in range(33)]
    lm[11] = _landmark(shoulder_x, SHOULDER_Y)   # left shoulder
    lm[12] = _landmark(shoulder_x, SHOULDER_Y)   # right shoulder
    lm[23] = _landmark(CENTRE_X, HIP_Y)          # left hip
    lm[24] = _landmark(CENTRE_X, HIP_Y)          # right hip
    lm[25] = _landmark(knee_x, KNEE_Y)           # left knee
    lm[26] = _landmark(knee_x, KNEE_Y)           # right knee
    lm[27] = _landmark(CENTRE_X, ANKLE_Y)        # left ankle
    lm[28] = _landmark(CENTRE_X, ANKLE_Y)        # right ankle
    return lm


def rep_angles(period_ms, bottom):
    """Knee angles for one repetition at the requested cadence.

    A repetition needs one frame below the descent threshold and one above the
    standing threshold, so the shortest possible cadence is two frame
    intervals. Longer cadences hold the bottom position for the extra frames.
    """
    frames = max(2, round(period_ms / FRAME_INTERVAL_MS))
    return [bottom] * (frames - 1) + [STANDING_KNEE]


def percentiles(values):
    if not values:
        return None
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p95": ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))],
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def _row(label, stats):
    if stats is None:
        return f"  {label:<18} no samples"
    return (f"  {label:<18} n={stats['n']:<4} min={stats['min']:>8.1f}  "
            f"median={stats['median']:>8.1f}  p95={stats['p95']:>8.1f}  "
            f"max={stats['max']:>8.1f}")


async def run(args):
    from websockets.asyncio.client import connect  # lazy: keeps pure logic testable

    results = {}          # rep number -> measurements
    sent_at = {}          # rep number -> client stamp of the frame that ended it
    faults_seen = 0

    async with connect(URL) as ws:
        await ws.send(json.dumps({"type": "reset", "exercise": "squat"}))

        async def receive():
            nonlocal faults_seen
            while True:
                try:
                    message = await ws.recv()
                except Exception:
                    return
                data = json.loads(message)
                now = time.time() * 1000.0

                if data.get("type") == "rep":
                    rep = data["rep"]
                    sent_at[rep] = data.get("client_ts")
                    results.setdefault(rep, {})
                    results[rep]["verdict_ms"] = now - data["client_ts"]
                    results[rep]["quality"] = data.get("quality")
                    results[rep]["faults"] = data.get("faults", [])
                    if data.get("faults"):
                        faults_seen += 1

                elif data.get("type") == "cue":
                    rep = data["rep"]
                    results.setdefault(rep, {})
                    timing = data.get("timing", {})
                    results[rep]["queue_ms"] = timing.get("queue_ms")
                    results[rep]["llm_ms"] = timing.get("llm_ms")
                    if sent_at.get(rep) is not None:
                        results[rep]["cue_ms"] = now - sent_at[rep]

        reader = asyncio.create_task(receive())

        started = time.perf_counter()
        for _ in range(args.reps):
            for knee in rep_angles(args.period, args.bottom):
                await ws.send(json.dumps({
                    "type": "frame",
                    "exercise": "squat",
                    "client_ts": time.time() * 1000.0,
                    "landmarks": pose(knee, args.trunk),
                }))
                await asyncio.sleep(FRAME_INTERVAL_MS / 1000.0)
        elapsed = time.perf_counter() - started

        # Wait for cues still in the queue behind the model.
        deadline = time.perf_counter() + args.drain
        while time.perf_counter() < deadline:
            pending = [r for r in results if "cue_ms" not in results[r]]
            if len(results) >= args.reps and not pending:
                break
            await asyncio.sleep(0.1)

        reader.cancel()

    counted = max(results) if results else 0
    verdicts = [v["verdict_ms"] for v in results.values() if "verdict_ms" in v]
    cues = [v["cue_ms"] for v in results.values() if "cue_ms" in v]
    queues = [v["queue_ms"]
              for v in results.values() if v.get("queue_ms") is not None]
    llms = [v["llm_ms"]
            for v in results.values() if v.get("llm_ms") is not None]

    print()
    print(f"Cadence            {args.period} ms per rep "
          f"({len(rep_angles(args.period, args.bottom))} frames at "
          f"{FRAME_INTERVAL_MS} ms)")
    print(f"Repetitions        expected {args.reps}, counted {counted}, "
          f"{'match' if counted == args.reps else 'MISMATCH'}")
    print(f"Send duration      {elapsed:.1f} s")
    print(f"Reps with a fault  {faults_seen} of {len(results)}")
    print("Latency, ms:")
    print(_row("verdict", percentiles(verdicts)))
    print(_row("cue", percentiles(cues)))
    print(_row("queue wait", percentiles(queues)))
    print(_row("generation", percentiles(llms)))

    missing = sorted(r for r in results if "cue_ms" not in results[r])
    if missing:
        print(f"  cues not received within the drain window: {missing}")

    if args.out:
        with open(args.out, "w") as f:
            f.write("rep,verdict_ms,queue_ms,llm_ms,cue_ms,quality,faults\n")
            for rep in sorted(results):
                r = results[rep]
                f.write(f"{rep},{r.get('verdict_ms', '')},{r.get('queue_ms', '')},"
                        f"{r.get('llm_ms', '')},{r.get('cue_ms', '')},"
                        f"{r.get('quality', '')},"
                        f"{' '.join(r.get('faults', []))}\n")
        print(f"\nWritten to {args.out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reps", type=int, default=20,
                   help="number of repetitions to drive (default 20)")
    p.add_argument("--period", type=int, default=1000,
                   help="milliseconds per repetition; rounded to whole frames "
                        "of %d ms, minimum two (default 1000)" % FRAME_INTERVAL_MS)
    p.add_argument("--bottom", type=float, default=80.0,
                   help="knee angle at the bottom of each rep; below the 95 "
                        "degree target for a clean rep (default 80)")
    p.add_argument("--trunk", type=float, default=20.0,
                   help="trunk lean held through the rep; below the 50 degree "
                        "limit for a clean rep (default 20)")
    p.add_argument("--drain", type=float, default=30.0,
                   help="seconds to wait for outstanding cues (default 30)")
    p.add_argument("--out", help="write per-rep measurements to this CSV")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
