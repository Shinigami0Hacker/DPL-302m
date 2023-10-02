const videoElement = document.getElementById("camera");
const startButton = document.getElementById("startButton");
const stopButton = document.getElementById("stopButton");

let stream;
async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoElement.srcObject = stream;
    } catch (error) {
        console.error("Error accessing the camera:", error);
    }
}

function stopCamera() {
    if (stream) {
        const tracks = stream.getTracks();
        tracks.forEach((track) => {
            track.stop();
        });
        videoElement.srcObject = null;
    }
}

startButton.addEventListener("click", startCamera);
stopButton.addEventListener("click", stopCamera);