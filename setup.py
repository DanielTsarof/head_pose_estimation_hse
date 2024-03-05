from setuptools import setup, find_packages

setup(
    name="head_pose_estimator",
    version="0.1",
    packages=find_packages(exclude=["tests*"]),
)