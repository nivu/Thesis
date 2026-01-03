from setuptools import setup, find_packages

setup(
    name="yolo-3d",
    version="1.0.0",
    description="Real-time 3D object detection with YOLOv11 and Depth Anything v2",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        line.strip() 
        for line in open("requirements.txt", "r").readlines()
        if not line.startswith("#")
    ],
    python_requires=">=3.8"
)