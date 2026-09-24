import {
    PoseLandmarker,
    FilesetResolver,
    DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

const video = document.getElementById("webcam");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const exerciseSelect = document.getElementById("exercise-select");

// Landmarks are streamed at 15 Hz. The exercise recogniser was trained at
// that rate and needs every frame; the server samples every third frame for
// form analysis, preserving the 200 ms interval its thresholds were
// calibrated against.
const SEND_INTERVAL_MS = 1000 / 15;

let poseLandmarker = null;
let lastVideoTime = -1;
let lastSendTime = 0;

const socket = new WebSocket("ws://localhost:8765");

socket.addEventListener("open", () => {
    CoachUI.setStatus("live");
});

socket.addEventListener("error", () => {
    CoachUI.setStatus("offline");
    CoachUI.setCoaching("Backend not connected. Start the Python server.");
});

socket.addEventListener("close", () => {
    CoachUI.setStatus("offline", "Backend disconnected");
});

let lastCoaching = "";

function send(payload) {
    if (socket.readyState !== WebSocket.OPEN) return;
    socket.send(JSON.stringify(payload));
}

socket.addEventListener("message", (event) => {
    const data = JSON.parse(event.data);

    // Every message carries what the recogniser currently believes, so the
    // readout stays live whether or not a rep just completed.
    CoachUI.setDetection(data.detected, data.detect_confidence,
        data.mode, data.suppressed);

    // A completed repetition. Everything here was decided by the rule layer,
    // so it arrives as soon as the rep ends. The wording follows separately.
    if (data.type === "rep") {
        CoachUI.pushRep({
            rep: data.rep,
            exercise: data.exercise,
            faults: data.faults || [],
            metrics: data.metrics || {},
            quality: data.quality,
            partial: data.partial,
            latencyMs: data.latency_ms
        });
        return;
    }

    // The phrased sentence for a rep already on screen.
    if (data.type === "cue") {
        lastCoaching = data.coaching || "";
        CoachUI.applyCue(data.rep, data.coaching, data.latency_ms);
        return;
    }

    if (data.exercise) CoachUI.setExercise(data.exercise);
    if (data.coaching && data.coaching !== lastCoaching) {
        lastCoaching = data.coaching;
        CoachUI.setCoaching(data.coaching);
    }
});

// Switching mode rebuilds the analyser on the backend, which restarts the rep
// count. The panel is cleared here so both ends agree about the session.
if (exerciseSelect) {
    exerciseSelect.addEventListener("change", () => {
        CoachUI.reset();
        send({ type: "exercise", exercise: exerciseSelect.value });
    });
}

CoachUI.onReset = () => {
    send({ type: "reset", exercise: exerciseSelect ? exerciseSelect.value : "squat" });
};

async function createPoseLandmarker() {
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
    );
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath:
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numPoses: 1
    });
    CoachUI.setCoaching("Model loaded.");
}

async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480 },
            audio: false
        });
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictLoop);
    } catch (err) {
        CoachUI.setCoaching("Could not access camera. Please allow camera access and reload.");
        console.error(err);
    }
}

function predictLoop() {
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
    }

    if (poseLandmarker && video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime;
        const result = poseLandmarker.detectForVideo(video, performance.now());
        drawResult(result);
    }

    requestAnimationFrame(predictLoop);
}

let overlayTheme = null;
let overlayInk = null;

function overlayColours() {
    const theme = document.documentElement.getAttribute("data-theme");
    if (theme !== overlayTheme) {
        const cs = getComputedStyle(document.documentElement);
        overlayInk = {
            bone: cs.getPropertyValue("--ink").trim(),
            joint: cs.getPropertyValue("--good-ink").trim()
        };
        overlayTheme = theme;
    }
    return overlayInk;
}

function drawResult(result) {
    const utils = new DrawingUtils(ctx);
    const c = overlayColours();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (result.landmarks) {
        for (const landmarks of result.landmarks) {
            utils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, { color: c.bone, lineWidth: 2 });
            utils.drawLandmarks(landmarks, { radius: 3, color: c.joint });
            maybeSendLandmarks(landmarks);
        }
        if (result.landmarks.length) CoachUI.hideHint();
    }
}

function maybeSendLandmarks(landmarks) {
    const now = performance.now();
    if (now - lastSendTime < SEND_INTERVAL_MS) return;
    if (socket.readyState !== WebSocket.OPEN) return;
    lastSendTime = now;

    const payload = landmarks.map(p => ({
        x: p.x, y: p.y, z: p.z, visibility: p.visibility
    }));
    // The wall-clock stamp travels with the frame and comes back on both the
    // verdict and the cue, which is what makes the two stages separately
    // measurable rather than inferred from the backend's own timings.
    send({
        type: "frame",
        exercise: exerciseSelect.value,
        client_ts: Date.now(),
        landmarks: payload
    });
}

createPoseLandmarker().catch(err => {
    CoachUI.setCoaching("Model failed to load: " + err.message);
    console.error(err);
});
startCamera();