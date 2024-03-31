from setuptools import setup, find_packages


setup(
    name="bigearthnet",
    version="0.0.1",
    packages=find_packages(include=["bigearthnet", "bigearthnet.*"]),
    python_requires=">=3.8",
    install_requires=[
        "gdown==4.4.0",
        "gitpython==3.1.29",
        "hub==3.0.0",
        "hydra-core==1.2.0",
        "jupyter",
        "matplotlib==3.2.2",
        "numpy==1.23",
        "pyyaml==6.0",
        "pytest==3.6.4",
        "pytorch_lightning==1.6.4",
        "scikit-learn==1.0.2",
        "timm==0.6.11",
        "torch==1.11",
        "torch_tb_profiler==0.4.0",
        "tqdm==4.64.1",
    ],
    extras_require={
        "dev": ["opencv-python"],
    },
)
