from setuptools import setup, find_packages

setup(
    name="flowmatching",
    version="0.1.0",
    description="A pure PyTorch implementation of Conditional Flow Matching (CFM)",
    author="Sonn",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "numpy",
        "matplotlib",
        "scipy",
        "tqdm",
    ],
    python_requires=">=3.8",
)
