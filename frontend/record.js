import {
    PoseLandmarker,
    FilesetResolver,
    DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

/* Landmark recorder for the action-recognition dataset.

   Separate from the coaching app: it never opens a WebSocket or analyses
   form. It saves raw MediaPipe landmarks with a class label and the
   recording conditions, one file per session, so the dataset can be split by
   session. Capture runs at CAPTURE_HZ, the rate the live app streams at. */

const CAPTURE_HZ = 15;
const CAPTURE_INTERVAL_MS = 1000 / CAPTURE_HZ;

const video = document.getElementById("webcam");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");

const el = {
    label: document.getElementById("label"),
    distance: document.getElementById("distance"),
    lighting: document.getElementById("lighting"),
    notes: document.getElementById("notes"),
    record: document.getElementById("record"),
    state: document.getElementById("state"),
    duration: document.getElementById("duration"),
    frames: document.getElementById("frames"),
    rate: document.getElementById("rate"),
    detected: document.getElementById("detected"),
    list: document.getElementById("session-list")
};

let poseLandmarker = null;
let lastVideoTime = -1;
let latestLandmarks = null;

let recording = false;
let session = null;
let captureTimer = null;

function stamp() {
    const d = new Date();
    const p = (n) => String(n).padStart(2, "0");
    return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}`
        + `-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}

function round4(n) {
    return Math.round(n * 10000) / 10000;
}

function startRecording() {
    session = {
        schema: "calisthenics-recording/1",
        session_id: `${el.label.value}-${stamp()}`,
        label: el.label.value,
        distance: el.distance.value,
        lighting: el.lighting.value,
        notes: el.notes.value || "",
        landmark_layout: "mediapipe-33",
        capture_hz_target: CAPTURE_HZ,
        started_at: new Date().toISOString(),
        frames: []
    };

    recording = true;
    const t0 = performance.now();

    // A fixed-interval timer rather than the render loop: the capture rate
    // must be the intended one, not whatever the camera happens to deliver.
    captureTimer = setInterval(() => {
        if (!latestLandmarks) return;
        session.frames.push({
            t: Math.round(performance.now() - t0),
            lm: latestLandmarks.map(p => [
                round4(p.x), round4(p.y), round4(p.z),
                round4(p.visibility ?? 0)
            ])
        });
        const seconds = (performance.now() - t0) / 1000;
        el.duration.textContent = seconds.toFixed(1) + " s";
        el.frames.textContent = session.frames.length;
        el.rate.textContent = (session.frames.length / seconds).toFixed(1) + " Hz";
    }, CAPTURE_INTERVAL_MS);

    el.record.textContent = "Stop and save";
    el.record.dataset.state = "on";
    el.state.textContent = "recording " + session.label;
}

function stopRecording() {
    clearInterval(captureTimer);
    recording = false;

    const seconds = session.frames.length / CAPTURE_HZ;
    session.duration_s = Math.round(seconds * 10) / 10;
    session.frame_count = session.frames.length;

    if (session.frame_count === 0) {
        el.state.textContent = "discarded, no frames";
    } else {
        download(session);
        const li = document.createElement("li");
        li.textContent = `${session.session_id} · ${session.frame_count} frames`
            + ` · ${session.distance}/${session.lighting}`;
        el.list.appendChild(li);
        el.state.textContent = "saved";
    }

    el.record.textContent = "Start recording";
    el.record.dataset.state = "off";
    session = null;
}

function download(data) {
    const blob = new Blob([JSON.stringify(data)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = data.session_id + ".json";
    a.click();
    URL.revokeObjectURL(url);
}

el.record.addEventListener("click", () => {
    if (recording) stopRecording(); else startRecording();
});

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
    el.record.disabled = false;
    el.record.textContent = "Start recording";
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
        el.state.textContent = "no camera access";
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
        draw(result);
    }

    requestAnimationFrame(predictLoop);
}

function draw(result) {
    const utils = new DrawingUtils(ctx);
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (result.landmarks && result.landmarks.length) {
        const landmarks = result.landmarks[0];
        latestLandmarks = landmarks;
        utils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS,
            { color: "#9a9c90", lineWidth: 2 });
        utils.drawLandmarks(landmarks, { radius: 3, color: "#7fb096" });
        el.detected.textContent = "yes";
    } else {
        latestLandmarks = null;
        el.detected.textContent = "no";
    }
}

createPoseLandmarker().catch(err => {
    el.state.textContent = "model failed to load";
    console.error(err);
});
startCamera();