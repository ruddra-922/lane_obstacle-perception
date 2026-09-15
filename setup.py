from setuptools import setup, find_packages

setup(
    name="lane-obstacle-perception",
    version="1.0.0",
    description="Vision-Based Autonomous Lane & Obstacle Perception Pipeline",
    author="Ruddraksh Dwivedi",
    author_email="ruddraksh.dwivedi2024@vitbhopal.ac.in",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.8.0",
        "numpy>=1.23.0",
        "pyyaml>=6.0",
        "matplotlib>=3.7.0"
    ],
    entry_points={
        "console_scripts": [
            "lane-perception=main:main"
        ]
    }
)
