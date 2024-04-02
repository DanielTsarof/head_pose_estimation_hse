from __future__ import annotations

import cv2
import numpy as np

from head_pose_estimator.src.head_pose import Processor


def test_processor_reg():
    processor = Processor(
        "head_pose_estimator/models/ssdlite_mobilenet_v2_face_300_integer_quant_with_postprocess.tflite",
        "head_pose_estimator/models/head_pose_estimator_integer_quant.tflite")
    path = "tests/test_image.png"
    image = cv2.imread(path)
    processed = processor.process_frame(image)
    test_processed = cv2.imread("tests/test_image_processed.png")
    print(type(image))
    # assert if there is the same image
    assert np.sum(processed - test_processed) == 0


def test_no_error():
    processor = Processor(
        "head_pose_estimator/models/ssdlite_mobilenet_v2_face_300_integer_quant_with_postprocess.tflite",
        "head_pose_estimator/models/head_pose_estimator_integer_quant.tflite")
    image_big = cv2.imread("tests/test_image.png")
    image_small = cv2.imread("tests/test_image_small.jpg")

    processed_big = processor.process_frame(image_big)
    processed_small = processor.process_frame(image_small)

    assert isinstance(processed_big, np.ndarray)
    assert isinstance(processed_small, np.ndarray)


if __name__ == "__main__":
    test_processor_reg()
    test_no_error()
