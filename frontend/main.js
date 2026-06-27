const video = document.getElementById("webcam");
const coachingText = document.getElementById("coaching-text");

async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480 },
            audio: false
        });
        video.srcObject = stream;
        coachingText.textContent = "Camera ready.";
    } catch (err) {
        coachingText.textContent = "Could not access camera. Please allow camera access and reload.";
        console.error(err);
    }
}

startCamera();