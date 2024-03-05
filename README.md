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

    cd models
    
    ./download.sh
    
    cd ..

Run streamlit demo:

    python run_demo.py

Run local demo:
    
    python main.py

### PIP

    pip install git+https://github.com/DanielTsarof/head_pose_estimation_hse#egg=head_pose_estimator

    import head_pose_estimator

To run main.py or using Processor and FaceDetector classes you have to specify model paths

### Docker demo
    
    git clone git@github.com:DanielTsarof/head_pose_estimation_hse.git

    cd head_pose_estimation_hse

    docker build -t streamlit_demo .

    docker run -p 8501:8501 --device /dev/video0 streamlit_demo

## Using

    from src.head_pose import Processor
    from src.head_poose_estimation import HeadPoseEstimator

    processor = Processor(model_face_detect_path.tflite, model_head_pose_path.tflite, num_threads=4)
    
    # image if np.tensor (BGR format)
    processed_image = processor.process_frame(image)