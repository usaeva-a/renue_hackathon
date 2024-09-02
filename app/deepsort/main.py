import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import os
from time import time
from datetime import timedelta
from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from tracking import *

directory_model = input("Введите директорию, в которой лежит модель Yolo: ")
directory_video = input("Введите директорию, в которой лежит видео: ")

model_path = os.path.join(directory_model, 'yolov10x_v2_4_best.pt')
video_path = os.path.join(directory_video, '31-03-2024-09%3A34%3A24.mp4')
output_video_path = os.path.join(directory_video, 'result_video.mp4')

CONFIDENCE_THRESHOLD = 0.8
MAX_AGE = 60
MAX_IOU_DISTANCE = 0.8

try:
    cap = cv2.VideoCapture(0)  # видео с камеры
except:
    cap = cv2.VideoCapture(video_path)  # записанное видео

cap = cv2.VideoCapture(video_path)
writer_deepsort = create_video_writer(cap, output_video_path)

model = YOLO(model_path)

tracker = DeepSort(max_age=MAX_AGE, max_iou_distance=MAX_IOU_DISTANCE)

cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

track_history = defaultdict(lambda: [])
start = time()
frame_count = 0

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        break

    detections = model(frame)[0]
    results = []

    for data in detections.boxes.data.tolist():

        confidence = data[4]

        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue

        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        class_id = int(data[5])

        results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])

    tracks = tracker.update_tracks(results, frame=frame)

    frame, track_results, track_path = get_track_results(frame, tracks, track_history, colors, CONFIDENCE_THRESHOLD)

    cv2.imshow(frame, 'video')
    writer_deepsort.write(frame)
    writer_deepsort.write(track_path)
    frame_count += 1
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
writer_deepsort.release()
cv2.destroyAllWindows()
stop = time()

with open('deepsort_res.txt', 'w') as file:
   json.dump(track_results, file)