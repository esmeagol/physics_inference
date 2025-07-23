from setuptools import setup, find_packages

setup(
    name="PureCV",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.5.0",
        "numpy>=1.19.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Computer Vision utilities for cue sports analysis",
    license="MIT",
    python_requires=">=3.7",
)
