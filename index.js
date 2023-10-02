pushUserInput = () => {

    let user_input = document.getElementById('user_input')
    let chat_section = document.getElementById('chat_section')
    let post = document.createElement("p")

    if (user_input.value === ""){
        return
    }

    post.textContent = user_input.value
    post.id = "post_chat"

    chat_section.appendChild(post);
    user_input.value = ""
}

processModel = () => {
    console.log("Hello world")
}

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
