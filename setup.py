"""
setup.py - Package installation configuration for EAGLE-i Processor

This file enables installation of the project as a Python package,
allowing imports like: from src.eaglei_modules import processing
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="eaglei-processor",
    version="1.0.0",
    description="EAGLE-i Outage and Weather Data Processing Pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="<repository-url>",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "geopandas>=0.9.0",
        "shapely>=1.7.0",
        "pymc>=4.0.0",
        "arviz>=0.11.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "requests>=2.25.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.5b0",
            "flake8>=3.9.0",
            "isort>=5.9.0",
            "pylint>=2.8.0",
        ],
    },
)
