import {
    PoseLandmarker,
    FilesetResolver,
    DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

const video = document.getElementById("webcam");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const exerciseSelect = document.getElementById("exercise-select");

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

let lastRepSeen = 0;
let lastCoaching = "";

// The backend may report faults as a list or as a flag object; both end up
// as a list of identifiers here.
function normaliseFaults(raw) {
    if (!raw) return [];
    if (Array.isArray(raw)) return raw;
    if (typeof raw === "object") return Object.keys(raw).filter(k => raw[k]);
    return [raw];
}

socket.addEventListener("message", (event) => {
    const data = JSON.parse(event.data);
    console.log("verdict payload", data);   // temporary

    const reps = typeof data.reps === "number" ? data.reps : null;

    // A completed repetition: one feed item, one graph point.
    if (reps !== null && reps > lastRepSeen) {
        lastRepSeen = reps;
        lastCoaching = data.coaching || "";
        CoachUI.pushRep({
            rep: reps,
            exercise: exerciseSelect.value,
            faults: normaliseFaults(data.faults),
            text: data.coaching || "",
            metrics: data.angles || data.metrics || {},
            latencyMs: data.latency_ms
        });
        return;
    }

    // Anything else is live status text, not a rep.
    if (data.coaching && data.coaching !== lastCoaching) {
        lastCoaching = data.coaching;
        CoachUI.setCoaching(data.coaching);
    }
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
    if (now - lastSendTime < 200) return;
    if (socket.readyState !== WebSocket.OPEN) return;
    lastSendTime = now;

    const payload = landmarks.map(p => ({
        x: p.x, y: p.y, z: p.z, visibility: p.visibility
    }));
    socket.send(JSON.stringify({
        exercise: exerciseSelect.value,
        landmarks: payload
    }));
}

createPoseLandmarker().catch(err => {
    CoachUI.setCoaching("Model failed to load: " + err.message);
    console.error(err);
});
startCamera();