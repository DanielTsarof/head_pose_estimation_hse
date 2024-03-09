from __future__ import annotations

import math

import cv2
import numpy as np

from head_pose_estimator.src.head_pose_estimation import draw_annotation_box
from head_pose_estimator.src.head_pose_estimation import FaceDetector
from head_pose_estimator.src.head_pose_estimation import get_square_box
from head_pose_estimator.src.head_pose_estimation import HeadPoseEstimator
from head_pose_estimator.src.stabilizer import Stabilizer

detectfps = ""


class Processor:
    def __init__(self,
                 model_face_detect,
                 model_head_pose,
                 num_threads=4
                 ):

        self.model_face_detect = model_face_detect,
        self.model_head_pose = model_head_pose,
        self.num_threads = num_threads,

        # 3D model points.
        self.model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corner
            (-150.0, -150.0, -125.0),  # Mouth left corner
            (150.0, -150.0, -125.0)  # Mouth right corner
        ]) / 4.5

        raw_value = []
        with open('head_pose_estimator/models/model.txt') as file:
            for line in file:
                raw_value.append(line)
        self.model_points_68 = np.array(raw_value, dtype=np.float32)
        self.model_points_68 = np.reshape(self.model_points_68, (3, -1)).T
        # Transform the model into a front view.
        self.model_points_68[:, 2] *= -1

        # Introduce scalar stabilizers for pose.
        self.pose_stabilizers = [Stabilizer(state_num=2, measure_num=1, cov_process=0.1, cov_measure=0.1) for _ in
                                 range(6)]

        # Init Face Detector
        self.face_detector = FaceDetector(model_face_detect, num_threads)
        self.interpreter_face_detect = self.face_detector.interpreter_face_detect
        self.face_detector_input_details = self.face_detector.input_details
        self.face_detector_box = self.face_detector.box
        self.face_detector_scores = self.face_detector.scores
        self.face_detector_count = self.face_detector.count

        # Init Head Pose Estimator
        self.head_pose_estimator = HeadPoseEstimator(model_head_pose, num_threads)
        self.interpreter_head_pose = self.head_pose_estimator.interpreter_head_pose
        self.head_pose_estimator_input_details = self.head_pose_estimator.input_details
        self.head_pose_estimator_predictions = self.head_pose_estimator.predictions

    def process_frame(self, image):

        image_height = image.shape[0]
        image_width = image.shape[1]

        # Camera internals
        focal_length = image_width
        camera_center = (image_width / 2, image_height / 2)
        camera_matrix = np.array([[focal_length, 0, camera_center[0]], [0, focal_length, camera_center[1]], [0, 0, 1]],
                                 dtype="double")

        dist_coeefs = np.zeros((4, 1))

        # Rotation vector and translation vector
        r_vec = np.array([[0.01891013], [0.08560084], [-3.14392813]])
        t_vec = np.array([[-14.97821226], [-10.62040383], [-2053.03596872]])

        # Resize and normalize image for network input
        frame = cv2.resize(image, (300, 300))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.expand_dims(frame, axis=0)
        frame = frame.astype(np.float32)
        cv2.normalize(frame, frame, -1, 1, cv2.NORM_MINMAX)

        # run model - Face Detector
        self.interpreter_face_detect.set_tensor(self.face_detector_input_details, frame)
        self.interpreter_face_detect.invoke()

        # get results - Face Detector
        boxes = self.interpreter_face_detect.get_tensor(self.face_detector_box)[0]
        scores = self.interpreter_face_detect.get_tensor(self.face_detector_scores)[0]
        count = self.interpreter_face_detect.get_tensor(self.face_detector_count)[0]

        for i, (box, score) in enumerate(zip(boxes, scores)):
            probability = score
            if probability >= 0.6:
                if (not math.isnan(box[0]) and
                        not math.isnan(box[1]) and
                        not math.isnan(box[2]) and
                        not math.isnan(box[3])):
                    pass
                else:
                    continue
                ymin = int(box[0] * image_height)
                xmin = int(box[1] * image_width)
                ymax = int(box[2] * image_height)
                xmax = int(box[3] * image_width)
                if ymin > ymax:
                    continue
                if xmin > xmax:
                    continue

                offset_y = int(abs(ymax - ymin) * 0.1)
                facebox = get_square_box([xmin, ymin + offset_y, xmax, ymax + offset_y])
                if not (facebox[0] >= 0 and facebox[1] >= 0 and facebox[2] <= image_width and
                        facebox[3] <= image_height):
                    continue

                face_img = image[facebox[1]:facebox[3], facebox[0]:facebox[2]]
                face_img = cv2.resize(face_img, (128, 128))
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                face_img = np.expand_dims(face_img, axis=0)
                face_img = face_img.astype(np.float32)

                # run model - Head Pose
                self.interpreter_head_pose.set_tensor(self.head_pose_estimator_input_details, face_img)
                self.interpreter_head_pose.invoke()

                # get results - Head Pose
                predictions = self.interpreter_head_pose.get_tensor(self.head_pose_estimator_predictions)[0]
                marks = np.array(predictions).flatten()[:136]
                marks = np.reshape(marks, (-1, 2))

                # Convert the marks locations from local CNN to global image.
                marks *= (facebox[2] - facebox[0])
                marks[:, 0] += facebox[0]
                marks[:, 1] += facebox[1]

                # Try pose estimation with 68 points.
                if r_vec is None:
                    (_, rotation_vector, translation_vector) = cv2.solvePnP(self.model_points_68, marks, camera_matrix,
                                                                            dist_coeefs)
                    r_vec = rotation_vector
                    t_vec = translation_vector
                (_, rotation_vector, translation_vector) = cv2.solvePnP(self.model_points_68,
                                                                        marks,
                                                                        camera_matrix,
                                                                        dist_coeefs,
                                                                        rvec=r_vec,
                                                                        tvec=t_vec,
                                                                        useExtrinsicGuess=True)
                pose = (rotation_vector, translation_vector)

                # Stabilize the pose.
                steady_pose = []
                pose_np = np.array(pose).flatten()
                for value, ps_stb in zip(pose_np, self.pose_stabilizers):
                    ps_stb.update([value])
                    steady_pose.append(ps_stb.state[0])
                steady_pose = np.reshape(steady_pose, (-1, 3))

                # Draw boxes
                draw_annotation_box(image, steady_pose[0], steady_pose[1], camera_matrix, dist_coeefs,
                                    color=(128, 255, 128))
                cv2.putText(image,
                            detectfps,
                            (image_width - 170, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (38, 0, 255),
                            1,
                            cv2.LINE_AA)

            if i >= (count - 1):
                break
        return image

    def estimate_pose(self, image):
        pass


if __name__ == '__main__':
    processor = Processor("../models/ssdlite_mobilenet_v2_face_300_integer_quant_with_postprocess.tflite",
                          "../models/head_pose_estimator_integer_quant.tflite")
    path = "/home/dtsarev/Изображения/ex_im1.png"
    image = cv2.imread(path)
    processed = processor.process_frame(image)

    cv2.imshow("ex", processed)
    cv2.waitKey(0)
