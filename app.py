from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
import os
import uuid
from detector import generate_heatmap

app = Flask(__name__)

# Dossiers pour les fichiers uploadés et les résultats
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/webcam")
def webcam():
    return render_template("webcam.html")

@app.route("/upload_video", methods=["POST"])
def upload_video():
    video = request.files.get("video")
    json_file = request.files.get("json")

    if not video or not json_file:
        return "Video or JSON file missing", 400

    # Génère des noms de fichiers uniques
    video_filename = f"{uuid.uuid4()}.mp4"
    json_filename = f"{uuid.uuid4()}.json"

    video_path = os.path.join(UPLOAD_FOLDER, secure_filename(video_filename))
    json_path = os.path.join(UPLOAD_FOLDER, secure_filename(json_filename))

    video.save(video_path)
    json_file.save(json_path)

    # Appelle le module de détection (YOLO + heatmap)
    result_video_filename = generate_heatmap(video_path, json_path)
    video_path_for_template = f"results/{result_video_filename}"

    return render_template(
        "result.html",
        input_type="video",
        original_filename=video.filename,
        video_path=video_path_for_template,
        detection_details=None
    )

@app.route("/process_frame_webcam", methods=["POST"])
def process_webcam_frame():
    # Cette route peut être complétée plus tard pour du temps réel
    return jsonify({"detections": []})

if __name__ == "__main__":
    # Important pour exécution dans Docker
    app.run(debug=True, host="0.0.0.0", port=5000)
