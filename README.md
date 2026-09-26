# CM3070 Final Project: Real-Time Calisthenics Coaching

A real-time form-coaching system for bodyweight exercise, built on template 4.1
of the CM3020 Artificial Intelligence templates, "Project Idea 1: Orchestrating
AI models to achieve a goal".

The system orchestrates three pre-trained models across three data types:

| Model | Input | Role |
|---|---|---|
| MediaPipe Pose (BlazePose) | image pixels | extracts 33 body landmarks per frame, in the browser |
| ST-GCN, adapted from an NTU60 checkpoint | skeleton time-series | classifies a 2 s window as squat, lunge or other |
| Llama 3.1 8B via Ollama | text | phrases the verdict as one coaching sentence |

A rule-based layer sits between recognition and the language model. It counts
repetitions, measures joint angles, judges each repetition against calibrated
thresholds, scores it, and selects which faults to mention. The language model
only phrases that result and never decides whether form is correct. Raw video
never leaves the browser: only landmarks are sent to the backend.

Scope: bodyweight squat and forward lunge, side-on camera. Coaching support
only, not medical or clinical advice.

## Requirements

- Python 3 and pip
- [Ollama](https://ollama.com) with the `llama3.1:8b` model
- A webcam and a current desktop browser
- Internet access for the frontend, which loads MediaPipe from a CDN

## Setup

```bash
# Python dependencies
cd backend
pip install -r requirements.txt

# Language model
ollama pull llama3.1:8b
```

## Running

Start Ollama if it is not already running, then use two terminals.

Terminal 1, backend (WebSocket server on `ws://localhost:8765`):

```bash
cd backend
python3 server.py
```

Terminal 2, frontend (static server on port 8000):

```bash
cd frontend
python3 -m http.server
```

Open `http://localhost:8000` and allow camera access. Stand side-on to the
camera with your whole body in frame.

Choose **auto** to let the recogniser select the exercise, or choose squat or
lunge manually. The verdict for each repetition appears immediately and the
phrased cue follows it.

If PyTorch or the recogniser weights (`training/recogniser.pt`) are missing,
the server prints a warning and runs with manual exercise selection only. If
Ollama is unavailable, cues fall back to fixed wording.

## Tests

```bash
cd backend
python3 -m unittest discover tests -v
```

71 tests cover the squat and lunge analysers, quality scoring and the coaching
layer.

## Repository layout

```
frontend/
  index.html, main.js, ui.js, style.css   coaching interface
  record.html, record.js                  recorder for labelled training sessions
backend/
  server.py        WebSocket server; orchestrates the three models
  angles.py        joint angles and visibility gating
  squat.py         squat state machine and verdict
  lunge.py         lunge state machine and verdict
  quality.py       0 to 100 repetition score
  coach.py         fault selection and language-model phrasing
  recogniser.py    live recognition and the exercise gate
  load_client.py   synthetic client for latency and throughput tests
  pose_quality.py  share of recorded frames passing the analysers' checks
  audit_cues.py    faithfulness and coverage audit of logged cues
  tests/           unit tests
  evaluations/     session logs and results reported in the evaluation
training/
  adaptation of the pre-trained ST-GCN; see training/README.md
```

## Evaluation data

The CSV files in `backend/evaluations/` are the logs and outputs behind the
results in the final report: live sessions, synthetic load runs, pose-tracking
quality and the coaching-cue audit. To reproduce the latency runs, start the
backend and run for example:

```bash
cd backend
python3 load_client.py --reps 20 --period 1000
```

## Limitations

All thresholds and the recogniser were fitted to one experienced user under the
Version A ethics scope, so they should be treated as person-specific. See the
Evaluation chapter of the final report for the full results and limitations.