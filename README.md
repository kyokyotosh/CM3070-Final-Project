# CM3070 Final Project — Real-Time Calisthenics Coaching

A real-time form-coaching system for bodyweight exercise, built on
Template 4.1 (Orchestrating AI Models to Achieve a Goal).

The system orchestrates three pre-trained models across three data types:

- **Pose estimation** (MediaPipe Pose): operates on raw pixels
- **Action recognition**: operates on skeleton time-series
- **Language model** (Ollama): operates on text

A deterministic form-analysis layer sits between pose estimation and the
language model: it computes joint angles and diagnoses form, and the
language model only phrases that diagnosis as coaching feedback. The
language model does not decide whether form is correct.

Scope: squat plus forward lunge. Framed as coaching support, not medical
or clinical advice.

## Structure

- `frontend/`: browser-side capture and pose estimation
- `backend/`: WebSocket server, form analysis, coaching-text generation