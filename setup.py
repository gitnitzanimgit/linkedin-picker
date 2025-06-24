from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="linkedin-photo-optimizer",
    version="1.0.0",
    author="LinkedIn Photo Optimizer Team",
    description="AI-powered LinkedIn profile photo selection and enhancement",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/LinkedInPicker",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Graphics :: Graphics Conversion",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "linkedin-optimizer=main:main",
            "linkedin-enhance=enhance_single:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.onnx", "*.json"],
    },
)