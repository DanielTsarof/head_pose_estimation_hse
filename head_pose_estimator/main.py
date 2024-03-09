from typing import Union
import argparse
import time
import os

import cv2

from head_pose_estimator.src.head_pose import Processor


def run_process(model_face_detect,
                model_head_pose,
                image_width: int = 1280,
                image_height: int = 720,
                vidfps: int = 30,
                num_threads: int = 4,
                usbcamno: int = 0,
                vid_file: Union[str, None] = None,
                show: bool = True):
    """
    :param model_face_detect: path ot face detection model
    :param model_head_pose: path head pose estimation model
    :param image_width: video or camera width resolution
    :param image_height: video or camera height resolution
    :param vidfps: Frame rate
    :param num_threads: max number of threads to be used
    :param usbcamno: usb cam id (otional)
    :param vid_file: path to video file

    one of the params "usbcamo" or "vid_file" shoul be specified
    """
    detectframecount = 0
    framecount = 0
    time1 = 0
    time2 = 0

    processor = Processor(model_face_detect,
                          model_head_pose,
                          num_threads=num_threads)

    if vid_file:
        cam = cv2.VideoCapture(vid_file)
        cam.set(cv2.CAP_PROP_FPS, vidfps)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)
        window_name = "videofile"

        # Определить кодек и создать объект VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Или другой кодек, подходящий для вашего файла

        output_video_path = os.path.join(os.sep.join(vid_file.split(os.sep)[:-1]), 'processed_' + vid_file.split(os.sep)[-1])
        out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (int(cam.get(3)), int(cam.get(4))), isColor=True)
    else:
        # Init Camera
        cam = cv2.VideoCapture(usbcamno)
        cam.set(cv2.CAP_PROP_FPS, vidfps)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)
        window_name = "USB Camera"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    while True:
        start_time = time.perf_counter()

        ret, image = cam.read()
        if not ret:
            if vid_file:
                break
            else:
                continue

        processor.process_frame(image)
        if show:
            cv2.imshow('USB Camera', image)
        if vid_file:
            out.write(image)

        if vid_file and not cam.isOpened():
            cam.release()
            out.release()
            cv2.destroyAllWindows()
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        detectframecount += 1

        # FPS calculation
        framecount += 1
        if framecount >= 10:
            fps = "(Playback) {:.1f} FPS".format(time1 / 10)
            detectfps = "(Detection) {:.1f} FPS".format(detectframecount / time2)
            framecount = 0
            detectframecount = 0
            time1 = 0
            time2 = 0
        end_time = time.perf_counter()
        elapsedTime = end_time - start_time
        time1 += 1 / elapsedTime
        time2 += elapsedTime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_face_detect",
                        default="head_pose_estimator/models/ssdlite_mobilenet_v2_face_300_integer_quant_with_postprocess.tflite",
                        help="Path of the detection model.")
    parser.add_argument("--model_head_pose", default="head_pose_estimator/models/head_pose_estimator_integer_quant.tflite",
                        help="Path of the detection model.")
    parser.add_argument("--usbcamno", type=int, default=0, help="USB Camera number.")

    parser.add_argument("--vid", type=int, default=None, help="video file path")

    parser.add_argument("--camera_width", type=int, default=1280, help="width.")
    parser.add_argument("--camera_height", type=int, default=720, help="height.")
    parser.add_argument("--vidfps", type=int, default=30, help="Frame rate.")
    parser.add_argument("--num_threads", type=int, default=4, help="Threads.")
    args = parser.parse_args()

    model_face_detect = args.model_face_detect
    model_head_pose = args.model_head_pose
    usbcamno = args.usbcamno
    image_width = args.camera_width
    image_height = args.camera_height
    vidfps = args.vidfps
    num_threads = args.num_threads
    vid_file = args.vid

    run_process(model_face_detect,
                model_head_pose,
                image_width,
                image_height,
                vidfps,
                num_threads,
                usbcamno,
                vid_file,
                show=True)


if __name__ == "__main__":
    main()
