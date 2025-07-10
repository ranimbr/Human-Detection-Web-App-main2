document.addEventListener('DOMContentLoaded', function() {
    const videoElement = document.getElementById('videoElement');
    const canvasElement = document.getElementById('canvasElement');
    const canvasContext = canvasElement.getContext('2d');
    const startButton = document.getElementById('startButton');
    const stopButton = document.getElementById('stopButton');
    const recordButton = document.getElementById('recordButton');
    const stopRecordButton = document.getElementById('stopRecordButton');
    const outputDiv = document.getElementById('output');
    const themeToggle = document.getElementById('theme-toggle');
    const body = document.body;

    let stream = null;
    let mediaRecorder = null;
    let recordedChunks = [];
    let detectionInterval;
    let previousDetections = []; // Store detections from the previous frame

    startButton.addEventListener('click', async () => {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
            videoElement.srcObject = stream;
            startButton.disabled = true;
            stopButton.disabled = false;
            recordButton.disabled = false;
        } catch (error) {
            console.error('Error accessing webcam:', error);
            outputDiv.innerText = 'Error accessing webcam.';
        }
    });

    stopButton.addEventListener('click', () => {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            videoElement.srcObject = null;
            startButton.disabled = false;
            stopButton.disabled = true;
            recordButton.disabled = true;
            stopRecordButton.disabled = true;
            clearInterval(detectionInterval);
        }
    });

    recordButton.addEventListener('click', () => {
        if (stream) {
            recordedChunks = [];
            // Instead of recording from videoElement, record from canvasElement
            const canvasStream = canvasElement.captureStream();
            mediaRecorder = new MediaRecorder(canvasStream, { mimeType: 'video/webm;codecs=vp9' }); // You can try other mimeTypes
            mediaRecorder.ondataavailable = event => {
                if (event.data.size > 0) {
                    recordedChunks.push(event.data);
                }
            };
            mediaRecorder.onstop = () => {
                const blob = new Blob(recordedChunks, { type: 'video/webm' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'webcam_detection.webm';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                recordButton.disabled = false;
                stopRecordButton.disabled = true;
            };
            mediaRecorder.start();
            recordButton.disabled = true;
            stopRecordButton.disabled = false;
        }
    });

    stopRecordButton.addEventListener('click', () => {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            mediaRecorder.stop();
        }
    });

    function processFrame() {
        if (videoElement.videoWidth > 0 && videoElement.videoHeight > 0) {
            // First, draw the video frame onto the canvas
            canvasElement.width = videoElement.videoWidth;
            canvasElement.height = videoElement.videoHeight;
            canvasContext.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);

            const imageDataURL = canvasElement.toDataURL('image/jpeg');

            fetch('/process_frame_webcam', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ image: imageDataURL })
            })
            .then(response => response.json())
            .then(data => {
                const currentDetections = data.detections || [];
                const persistentDetections = [];

                currentDetections.forEach(currentDetection => {
                    const [cx, cy, cw, ch] = currentDetection.box;
                    const currentLabel = currentDetection.label;
                    let foundMatch = false;

                    previousDetections.forEach(previousDetection => {
                        const [px, py, pw, ph] = previousDetection.box;
                        const previousLabel = previousDetection.label;

                        if (currentLabel === previousLabel && isClose(cx, cy, cw, ch, px, py, pw, ph)) {
                            persistentDetections.push(currentDetection);
                            foundMatch = true;
                        }
                    });

                    if (!foundMatch) {
                        persistentDetections.push(currentDetection);
                    }
                });

                // Then, draw the detection boxes on the same canvas
                persistentDetections.forEach(detection => {
                    const [x, y, w, h] = detection.box;
                    canvasContext.strokeStyle = 'lime';
                    canvasContext.lineWidth = 2;
                    canvasContext.strokeRect(x, y, w, h);
                    canvasContext.fillStyle = 'lime';
                    canvasContext.font = '10px sans-serif';
                    canvasContext.fillText(`${detection.label}: ${parseFloat(detection.confidence).toFixed(2)}`, x, y - 5);
                });

                previousDetections = currentDetections;
            })
            .catch(error => {
                console.error('Error sending frame for detection:', error);
            });
        }
    }

    function isClose(x1, y1, w1, h1, x2, y2, w2, h2, distanceThreshold = 70, sizeDifferenceThreshold = 0.5) {
        // Calculate centers
        const centerX1 = x1 + w1 / 2;
        const centerY1 = y1 + h1 / 2;
        const centerX2 = x2 + w2 / 2;
        const centerY2 = y2 + h2 / 2;

        // Calculate the distance between centers
        const distance = Math.sqrt(Math.pow(centerX1 - centerX2, 2) + Math.pow(centerY1 - centerY2, 2));

        // Calculate the size difference ratio
        const area1 = w1 * h1;
        const area2 = w2 * h2;
        const sizeDifferenceRatio = Math.abs(area1 - area2) / Math.max(area1, area2);

        return distance < distanceThreshold && sizeDifferenceRatio < sizeDifferenceThreshold;
  }

    videoElement.addEventListener('loadedmetadata', () => {
        // Set canvas dimensions to match video dimensions
        canvasElement.width = videoElement.videoWidth;
        canvasElement.height = videoElement.videoHeight;
        detectionInterval = setInterval(processFrame, 700);
    });

    // Theme toggle functionality
    if (themeToggle) {
        themeToggle.addEventListener('click', () => {
            body.classList.toggle('light-theme');
            const isLightTheme = body.classList.contains('light-theme');
            localStorage.setItem('theme', isLightTheme ? 'light' : 'dark');
        });

        const savedTheme = localStorage.getItem('theme');
        if (savedTheme === 'light') {
            body.classList.add('light-theme');
        }
    }
});