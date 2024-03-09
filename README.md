# Head Pose Estimation Project

-----------------------------

## Description

Demo application to perform head pose estimation in real time

## Installation

### From git

    git clone git@github.com:DanielTsarof/head_pose_estimation_hse.git

    cd head_pose_estimation_hse

    python -m venv venv

    source venv/bin/activate

    pip install -r requirements.txt

Build project

    python3 –m build

### Run streamlit demo:

    python run_demo.py

### Run local demo:

    python head_pose_estimator/main.py

or

    head_pose

### PIP

    pip install git+https://github.com/DanielTsarof/head_pose_estimation_hse#egg=head_pose_estimator


### Docker demo

    git clone git@github.com:DanielTsarof/head_pose_estimation_hse.git

    cd head_pose_estimation_hse

    docker build -t streamlit_demo .

    docker run -p 8501:8501 --device /dev/video0 streamlit_demo

### Run tests

    pytest -q tests/test_regression.py

## Using for frames processing

    from head_pose_estimator.src.head_pose import Processor
    from head_pose_estimator.src.head_poose_estimation import HeadPoseEstimator

    # image if np.tensor (BGR format)
    processed_image = processor.process_frame(image)

### pre-commit setup

    pre-commit install
