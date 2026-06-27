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
    coachingText.textContent = "Model loaded. Stand back so your whole body is in frame.";
}

async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480 },
            audio: false
        });
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictLoop);
        coachingText.textContent = "Camera ready. Loading model...";
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
        }
    }
}

createPoseLandmarker();
startCamera();