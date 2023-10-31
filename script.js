const audioFileInput = document.getElementById('audioFileInput');
const audioPlayer = document.getElementById('audioPlayer');
const waveformCanvas = document.getElementById('waveform');
const waveformContext = waveformCanvas.getContext('2d');
const audioList = document.getElementById('compsList')
// const Waveform = require('wavefrom')

// Update the displayed value dynamically
document.getElementById("numComponentsSlider").addEventListener("input", function() {
    const numComponentsValue = document.getElementById("numComponentsValue");
    numComponentsValue.textContent = this.value;
});


audioFileInput.addEventListener("change", handleFileUpload);
function handleFileUpload() {
    // Clear previous waveform
    // waveformContext.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);

    console.log(audioFileInput.files)
    const showAudio = document.getElementById("audioDisplay");
    showAudio.style.display = showAudio.style.display === "none" ? "block" : "none";

    const file = audioFileInput.files[0];
    if (file) {
        // Set the audio source to the selected file
        audioPlayer.src = URL.createObjectURL(file);

        // Show the player panel
        showAudio.style.display = 'block';
        
        // Create a Waveform.js instance for the canvas
        // const waveform = new Waveform({
        //     container: waveformCanvas,
        //     interpolate: true,
        // });
        
        // Load and render the waveform
        // waveform.load(audioPlayer.src, () => {
        //     waveform.play();
        // });
    }
}


document.getElementById("uploadButton").addEventListener("click", () => {
    const fileInput = document.getElementById("audioFileInput");
    const file = fileInput.files[0];
    const numComps = document.getElementById("numComponentsSlider").value;

    if (file) {
        const formData = new FormData();
        formData.append("audio", file);

        // Make an HTTP POST request to the API
        fetch("http://127.0.0.1:5000/process_audio", {
            method: "POST",
            headers: {
            'Access-Control-Allow-Origin':'*'
             },
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            console.log(data);
            console.log("yesss")
            audioList.innerHTML = ""; 
            
                data.audios.forEach((audioObj, index) => {
                const audioElement = document.createElement("audio");
                audioElement.src = `data:audio/mpeg;base64,${audioObj.audio_data}`;
                audioElement.controls = true;
                audioList.appendChild(audioElement);

                const showGraphButton = document.createElement("button");
                showGraphButton.textContent = "Show Graph";
                showGraphButton.addEventListener("click", () => {
                    alert(`Showing graph for Component ${index + 1}`);
                });
                audioList.appendChild(showGraphButton);
                })
        })
        .catch(error => {
            console.error("Error: " + error);
            alert("An error occurred while uploading the file.");
        });
    } else {
        alert("Please select an audio file to upload.");
    }

});

document.getElementById("waveformButton").addEventListener("click", () => {
    // console.log("fn running");    
    const fileInput = document.getElementById("audioFileInput");
    const file = fileInput.files[0];

    if (file) {
        const formData = new FormData();
        formData.append("audio", file);

        // Make an HTTP POST request to the API
        fetch("http://127.0.0.1:5000/get_waveform", {
            method: "POST",
            headers: {
            'Access-Control-Allow-Origin':'*'
             },
            body: formData
        })
        .then(response => response.blob())
        .then(data => {
            console.log("File uploaded and sent successfully to Waveform endpoint!");

            const img = new Image();
            img.src = URL.createObjectURL(data);
                    
            img.onload = () => {
                waveformContext.drawImage(img, 0, 0, waveformCanvas.width, waveformCanvas.height);
            };
        })
        .catch(error => {
            console.error("Error: " + error);
            alert("An error occurred.");
        });
    } else {
        alert("Please select an audio file to upload.");
    }

});
