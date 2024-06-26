from __future__ import annotations

import cv2
import streamlit as st
from PIL import Image

from head_pose_estimator.src.head_pose import Processor

processor = Processor("head_pose_estimator/models/ssdlite_mobilenet_v2_face_300_integer_quant_with_postprocess.tflite",
                      "head_pose_estimator/models/head_pose_estimator_integer_quant.tflite")


def main():
    st.title("Head Pose Estimation Demo")
    st.write("Head pose estimation project webcam live demo using OpenCV and Streamlit.")

    # Создаем плейсхолдер для отображения изображения
    frame_holder = st.empty()

    # Захват видео с первой веб-камеры
    cap = cv2.VideoCapture(0)
    while True:
        # Читаем кадр из видеопотока
        ret, frame = cap.read()
        if not ret:
            break

        # Обработка изображения
        processed_frame = processor.process_frame(frame)

        # Конвертируем изображение обратно в формат, который можно отобразить в Streamlit
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(processed_frame)

        # Отображаем обработанное изображение
        frame_holder.image(img)

        # Если нужно остановить цикл (например, по нажатию кнопки в Streamlit)

    cap.release()


if __name__ == "__main__":
    main()
