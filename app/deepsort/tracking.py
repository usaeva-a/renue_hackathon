from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort

def create_video_writer(cap, output_filename):
    """
    Создаёт объект для записи видео в нужном формате.

        Параметры:
            cap - объект VideoCapture,
            output_filename (str) - название выходного файла.
        Возвращает:
            writer - объект для записи видео
    """
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
    writer = cv2.VideoWriter(output_filename, fourcc, fps,
                             (frame_width, frame_height))

    return writer

def get_track_results(frame, tracks, track_history, colors, CONFIDENCE_THRESHOLD):
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

    return frame, track_results, track_path
