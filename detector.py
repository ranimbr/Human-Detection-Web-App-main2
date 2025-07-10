import mlflow  # à mettre en haut du fichier detector.py
import cv2
import numpy as np
import json
import os
from shapely.geometry import Polygon, Point
from ultralytics import YOLO
from tqdm import tqdm
import matplotlib.pyplot as plt

def generate_heatmap(video_path, json_path, output_dir="static/results"):
    # Définit explicitement l'URI du tracking MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5001")

    mlflow.start_run()  # Démarrer une expérience MLflow

    os.makedirs(output_dir, exist_ok=True)

    mlflow.log_param("video_name", os.path.basename(video_path))
    mlflow.log_param("json_name", os.path.basename(json_path))
    mlflow.log_param("heatmap_radius", 20)

    with open(json_path, 'r') as f:
        data = json.load(f)

    polygons = []
    for ann in data['annotations']:
        if ann['type'] == 'polygon':
            pts = np.array(ann['points'], dtype=np.int32)
            polygons.append(pts)

    polygons_shapely = [Polygon(poly) for poly in polygons]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Erreur d'ouverture de la vidéo.")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_filename = "output_with_heatmap_and_countsf.mp4"
    output_path = os.path.join(output_dir, output_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    if not out.isOpened():
        raise IOError("Erreur : VideoWriter n'a pas pu s'ouvrir.")

    model = YOLO('yolov8n.pt')

    accumulated_heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)
    heatmap_radius = 20
    history = [[] for _ in range(len(polygons_shapely))]

    for frame_idx in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            print(f"Fin de la vidéo à la frame {frame_idx}")
            break

        results = model(frame)[0]
        person_centers = []

        for box in results.boxes.data.tolist():
            x1, y1, x2, y2, score, cls = box
            if int(cls) == 0:
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                person_centers.append((cx, cy))

        zone_counts = []
        zone_centers = []

        for i, poly in enumerate(polygons_shapely):
            count = sum(poly.contains(Point(x, y)) for x, y in person_centers)
            zone_counts.append(count)
            centroid = poly.centroid
            zone_centers.append((int(centroid.x), int(centroid.y)))
            history[i].append(count)

        heatmap_frame = np.zeros((frame_height, frame_width), dtype=np.float32)
        for (x, y) in person_centers:
            if 0 <= y < frame_height and 0 <= x < frame_width:
                cv2.circle(heatmap_frame, (x, y), heatmap_radius, 1.0, -1)
                cv2.circle(accumulated_heatmap, (x, y), heatmap_radius, 1.0, -1)

        heatmap_blur = cv2.GaussianBlur(heatmap_frame, (0, 0), sigmaX=15, sigmaY=15)
        heatmap_uint8 = cv2.normalize(heatmap_blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        accumulated_blur = cv2.GaussianBlur(accumulated_heatmap, (0, 0), sigmaX=25, sigmaY=25)
        accumulated_uint8 = cv2.normalize(accumulated_blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        accumulated_colormap = cv2.applyColorMap(accumulated_uint8, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(frame, 0.6, accumulated_colormap, 0.4, 0)

        for i, poly in enumerate(polygons):
            cv2.polylines(overlay, [poly], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(
                overlay,
                f"Zone {i+1}: {zone_counts[i]} pers",
                (zone_centers[i][0] - 50, zone_centers[i][1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        if frame_idx == frame_count - 1:
            cv2.imwrite(os.path.join(output_dir, "last_frame_with_heatmap.png"), overlay)

        out.write(overlay)

    cap.release()
    out.release()

    plt.figure(figsize=(12, 6))
    for i, zone_history in enumerate(history):
        plt.plot(zone_history, label=f"Zone {i+1}")
    plt.legend()
    plt.title("Evolution du nombre de personnes par zone")
    plt.xlabel("Frame")
    plt.ylabel("Nombre de personnes")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "zone_counts_evolutionf.png"))

    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    for poly in polygons:
        cv2.fillPoly(mask, [poly], 255)

    masked_heatmap = cv2.bitwise_and(accumulated_colormap, accumulated_colormap, mask=mask)
    cv2.imwrite(os.path.join(output_dir, "heatmap_cumulative_maskedf.png"), masked_heatmap)

    # --- MLflow logging ---
    mlflow.log_artifact(output_path)
    mlflow.log_artifact(os.path.join(output_dir, "zone_counts_evolutionf.png"))
    mlflow.log_artifact(os.path.join(output_dir, "heatmap_cumulative_maskedf.png"))
    mlflow.log_artifact(os.path.join(output_dir, "last_frame_with_heatmap.png"))
    mlflow.end_run()

    print("✅ Heatmap and results saved in:", output_dir)

    return output_filename

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python detector.py <chemin_video> <chemin_json>")
        sys.exit(1)

    video_path = sys.argv[1]
    json_path = sys.argv[2]
    print(f"Lancement du traitement pour {video_path} et {json_path}")

    output_file = generate_heatmap(video_path, json_path)
    print(f"Fichier de sortie généré : {output_file}")
