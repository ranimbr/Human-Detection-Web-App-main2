import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from detector import generate_heatmap

def test_generate_heatmap():
    video_path = "tests/sample.mp4"
    json_path = "tests/zones.json"

    assert os.path.exists(video_path), f"Fichier vidéo manquant: {video_path}"
    assert os.path.exists(json_path), f"Fichier JSON manquant: {json_path}"

    result_filename = generate_heatmap(video_path, json_path)
    output_path = os.path.join("static/results", result_filename)
    assert os.path.exists(output_path), f"Fichier de sortie non généré: {output_path}"
