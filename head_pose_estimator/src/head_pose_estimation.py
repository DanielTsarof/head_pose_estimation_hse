import cv2
import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except:
    from tensorflow.lite.python.interpreter import Interpreter


class FaceDetector:
    def __init__(self, model_face_detect: str, num_threads: int):
        # Init Face Detector
        self.interpreter_face_detect = Interpreter(model_path=model_face_detect)
        try:
            self.interpreter_face_detect.set_num_threads(num_threads)
        except:
            print("WARNING: The installed PythonAPI of Tensorflow/Tensorflow Lite runtime does not support Multi-Thread processing.")
            print("WARNING: It works in single thread mode.")
            print("WARNING: If you want to use Multi-Thread to improve performance on aarch64/armv7l platforms, please refer to one of the below to implement a customized Tensorflow/Tensorflow Lite runtime.")
            print("https://github.com/PINTO0309/Tensorflow-bin.git")
            print("https://github.com/PINTO0309/TensorflowLite-bin.git")
            pass
        self.interpreter_face_detect.allocate_tensors()
        self.input_details = self.interpreter_face_detect.get_input_details()[0]['index']
        self.box = self.interpreter_face_detect.get_output_details()[0]['index']
        self.scores = self.interpreter_face_detect.get_output_details()[2]['index']
        self.count = self.interpreter_face_detect.get_output_details()[3]['index']


class HeadPoseEstimator:
    def __init__(self, model_head_pose: str, num_threads: int):
        # Init Head Pose Estimator
        self.interpreter_head_pose = Interpreter(model_path=model_head_pose)
        try:
            self.interpreter_head_pose.set_num_threads(num_threads)
        except:
            pass
        self.interpreter_head_pose.allocate_tensors()
        self.input_details = self.interpreter_head_pose.get_input_details()[0]['index']
        self.predictions = self.interpreter_head_pose.get_output_details()[0]['index']


def get_square_box(box):
    """Get a square box out of the given box, by expanding it."""
    left_x = box[0]
    top_y = box[1]
    right_x = box[2]
    bottom_y = box[3]

    box_width = right_x - left_x
    box_height = bottom_y - top_y

    # Check if box is already a square. If not, make it a square.
    diff = box_height - box_width
    delta = int(abs(diff) / 2)
    if diff == 0:  # Already a square.
        return box
    elif diff > 0:  # Height > width, a slim box.
        left_x -= delta
        right_x += delta
        if diff % 2 == 1:
            right_x += 1
    else:  # Width > height, a short box.
        top_y -= delta
        bottom_y += delta
        if diff % 2 == 1:
            bottom_y += 1
    return [left_x, top_y, right_x, bottom_y]


def draw_annotation_box(image,
                        rotation_vector,
                        translation_vector,
                        camera_matrix,
                        dist_coeefs,
                        color=(255, 255, 255),
                        line_width=2):
    """Draw a 3D box as annotation of pose"""
    point_3d = []
    rear_size = 75
    rear_depth = 0
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = 100
    front_depth = 100
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float64).reshape(-1, 3)

    # Map to 2d image points
    (point_2d, _) = cv2.projectPoints(point_3d, rotation_vector, translation_vector, camera_matrix, dist_coeefs)
    point_2d = np.int32(point_2d.reshape(-1, 2))

    # Draw all the lines
    cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[1]), tuple(point_2d[6]), color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[2]), tuple(point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[3]), tuple(point_2d[8]), color, line_width, cv2.LINE_AA)
