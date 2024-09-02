import cv2

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