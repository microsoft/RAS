import os
import re
import sys

from setuptools import Command, find_packages, setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="ras",
    version="0.1",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="diffusion scheduler pytorch",
    license="Apache 2.0 License",
    author="Ziming Liu, Yifan Yang, Chengruidong Zhang, Yiqi Zhang, Lili Qiu, Yang You, Yuqing Yang",
    author_email="liuziming@comp.nus.edu.sg",
    url="https://github.com/MaruyamaAya/RAS",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=required,
    python_requires=">=3.12",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Environment :: GPU :: NVIDIA CUDA",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ]
)
