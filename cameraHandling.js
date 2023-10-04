const videoElement = document.getElementById("camera");

let stream;

async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoElement.srcObject = stream;
    } catch (error) {
        console.error("Error accessing the camera:", error);
    }
}

stopCamera = () => {
    if (stream) {
        const tracks = stream.getTracks();
        tracks.forEach((track) => {
            track.stop();
        });
        videoElement.srcObject = null;
    }
}

storeVideo = () => {
    return
}

