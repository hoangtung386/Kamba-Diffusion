from setuptools import setup, find_packages

setup(
    name="dmk_stroke",
    version="0.1.0",
    author="DMK Team",
    author_email="example@email.com",
    description="Diffusion Mamba-KAN for Stroke Segmentation",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/dmk_stroke",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy",
        "timm",
        "einops",
        "opencv-python",
        "matplotlib",
        "tqdm",
        "wandb",
        "scipy",
        # "mamba-ssm", # Note: Mamba often requires manual install
        # "efficient-kan", # Optional if using library implementation
    ],
)
