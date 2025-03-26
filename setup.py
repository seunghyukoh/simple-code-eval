from setuptools import setup, find_packages

setup(
    name="simple-code-eval",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pytest>=7.0.0",
    ],
    author="Seunghyuk Oh",
    author_email="seunghyukoh0@gmail.com",
    description="A simple code evaluation package",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/seunghyukoh/simple-code-eval",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
