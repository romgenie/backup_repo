"""
Setup script for the reliability extension for OpenAI Agents SDK.
"""

from setuptools import setup, find_packages

setup(
    name="openai-agents-reliability",
    version="0.1.0",
    description="Reliability extension for OpenAI Agents SDK to reduce hallucinations",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "openai-agents",
        "pydantic",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)
