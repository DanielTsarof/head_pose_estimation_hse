from setuptools import setup, find_packages

setup(
    name="head_pose_estimator",
    version="0.1",
    packages=find_packages(exclude=["tests*", "src.__pycache__*"]),
    package_data={
        "": ["*.md", "*.txt", "*.sh", "*.tflite", "head_pose_estimator/run_demo.py", "head_pose_estimator/main.py"],
        "models": ["*.sh", "*.tflite"],  # Явно указывает включение файлов в директории models
    },
    exclude_package_data={
        "": ["__pycache__", "*.pyc"],  # Исключает __pycache__ и скомпилированные Python файлы
    },
    entry_points={
        'console_scripts': [
            'head_pose=head_pose_estimator.main:main',
        ],
    },
    install_requires=[
    ],
    data_files=[('bin', ['head_pose_estimator/run_demo.py', 'head_pose_estimator/main.py'])]
)
