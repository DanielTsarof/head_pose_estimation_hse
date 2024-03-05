import cv2
import streamlit as st
from PIL import Image

from src.head_pose import Processor

processor = Processor("models/ssdlite_mobilenet_v2_face_300_integer_quant_with_postprocess.tflite",
                      "models/head_pose_estimator_integer_quant.tflite")


def main():
    st.title("Head Pose Estimation Demo")
    st.write("This is a simple webcam live feed demo using OpenCV and Streamlit.")

    # Создаем плейсхолдер для отображения изображения
    frame_holder = st.empty()

    # Захват видео с первой веб-камеры
    cap = cv2.VideoCapture(0)
    button1 = st.button('Stop')
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
        if button1:
            break

    cap.release()


if __name__ == "__main__":
    main()
