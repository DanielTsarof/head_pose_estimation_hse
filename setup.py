from setuptools import setup, find_packages

setup(
    name="head_pose_estimator",
    version="0.1",
    packages=find_packages(exclude=["tests*"]),
    entry_points={
        'console_scripts': [
            'head_pose=head_pose_estimator.main:main',  # package_name это команда, main.py - модуль, main - функция
        ],
    },
)