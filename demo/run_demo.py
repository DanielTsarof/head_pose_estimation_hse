import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Функция для обработки изображения (например, преобразование в градации серого)
def process_image(image):
    # Здесь можно применить любую обработку, например, преобразование в градации серого
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def main():
    st.title("Webcam Live Feed")
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
        processed_frame = process_image(frame)

        # Конвертируем изображение обратно в формат, который можно отобразить в Streamlit
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
        img = Image.fromarray(processed_frame)

        # Отображаем обработанное изображение
        frame_holder.image(img)

        # Если нужно остановить цикл (например, по нажатию кнопки в Streamlit)
        if button1:
            break

    cap.release()

if __name__ == "__main__":
    main()
