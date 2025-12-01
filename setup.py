"""
Self-Alignment Learning (SAL)
Communication-Based AI Growth

Training as dialogue, not control.
"""

from setuptools import setup, find_packages

setup(
    name="sal-learning",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.20.0",
    ],
    python_requires=">=3.8",
    author="Aaron Liam Lee",
    author_email="info@emergenzwerke.de",
    description="Self-Alignment Learning: Communication-Based AI Growth",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Whiteroom-Ai/Self-Alignment-Learning",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
