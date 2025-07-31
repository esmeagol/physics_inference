from setuptools import setup, find_packages

setup(
    name="snooker_inference",
    version="0.1",
    packages=find_packages(),
    py_modules=[
        'inference_runner',
        'local_pt_inference',
        'roboflow_local_inference',
        'tracker',
        'tracker_benchmark',
        'tracking'
    ],
    package_dir={'': '.'},
    install_requires=[
        'numpy',
        'opencv-python',
        'torch',
        'torchvision',
    ],
    python_requires='>=3.6',
)
