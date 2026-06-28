import {
    PoseLandmarker,
    FilesetResolver,
    DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

const video = document.getElementById("webcam");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const coachingText = document.getElementById("coaching-text");

let poseLandmarker = null;
let lastVideoTime = -1;
let lastSendTime = 0;

const socket = new WebSocket("ws://localhost:8765");

socket.addEventListener("open", () => {
    console.log("Connected to backend");
});

socket.addEventListener("error", () => {
    coachingText.textContent = "Backend not connected. Start the Python server.";
});

socket.addEventListener("message", (event) => {
    const data = JSON.parse(event.data);
    coachingText.textContent = data.coaching;
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
    coachingText.textContent = "Model loaded.";
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
        coachingText.textContent = "Could not access camera. Please allow camera access and reload.";
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

function drawResult(result) {
    const utils = new DrawingUtils(ctx);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (result.landmarks) {
        for (const landmarks of result.landmarks) {
            utils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, { color: "rgb(0, 255, 0)", lineWidth: 2 });
            utils.drawLandmarks(landmarks, { radius: 3, color: "rgb(255, 0, 0)" });
            maybeSendLandmarks(landmarks);
        }
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
    socket.send(JSON.stringify({ landmarks: payload }));
}

createPoseLandmarker().catch(err => {
    coachingText.textContent = "Model failed to load: " + err.message;
    console.error(err);
});
startCamera();