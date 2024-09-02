# need to install deepsort-realtime (https://github.com/levan92/deep_sort_realtime/ or pip install deep-sort-realtime),
# ultralytics (pip install ultralytics)

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

from write_video import create_video_writer

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

writer_deepsort = create_video_writer(cap, output_video_path)

model = YOLO(model_path)

tracker = DeepSort(max_age=MAX_AGE, max_iou_distance=MAX_IOU_DISTANCE)

cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

track_history = defaultdict(lambda: [])
start = time()

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

    track_results = {'track_id': [], 
                 'xmin': [], 'ymin': [], 
                 'xmax': [], 'ymax': [], 
                 'cx': [], 'cy': [],
                 'class_id': []}
    

    for track in tracks:
        track_id = track.track_id
        tlbr = track.to_tlbr(orig=True)
        score = track.det_conf
        class_id = track.det_class

        if score is None or score < CONFIDENCE_THRESHOLD:
            continue
     
        xmin, ymin, xmax, ymax = int(tlbr[0]), int(tlbr[1]), int(tlbr[2]), int(tlbr[3])
        cx, cy = int((xmin+xmax)/2), int((ymin+ymax)/2)

        track_results['track_id'].append(track_id)
        track_results['xmin'].append(xmin)
        track_results['ymin'].append(ymin)
        track_results['xmax'].append(xmin)
        track_results['ymax'].append(ymin)
        track_results['cx'].append(cx)
        track_results['cy'].append(cy)        
        track_results['class_id'].append(class_id)

        color = colors[int(track_id) % len(colors)]
        color = [idx * 255 for idx in color]
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 3)
        cv2.rectangle(frame, (xmin, ymin-30), (xmax, ymin), color, -1)
        cv2.putText(frame, 
                "track_id: " + str(track_id) + "-" + "class " + str(class_id),
                (xmin, ymin-10), 0, 0.75, (255,255,255), 2)
        
        track_info = track_history[track]
        track_info.append((cx, cy))
        if len(track_info) > 30:
            track_info.pop(0)

        points = np.hstack(track_info).astype(np.int32).reshape((-1, 1, 2))
        track_path = cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
 
    cv2.imshow(frame, 'video')
    writer_deepsort.write(frame)
    writer_deepsort.write(track_path)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
writer_deepsort.release()
cv2.destroyAllWindows()
stop = time()

print(f'''Inference stats:
      \tFrames: {idx}
      \tTime inference per frame: {(timedelta(seconds=stop-start).total_seconds()/idx)*1000:.2f} msec
      \tFull inference time: {timedelta(seconds=stop-start)}''')

with open('deepsort_res.txt', 'w') as file:
   json.dump(track_results, file)
