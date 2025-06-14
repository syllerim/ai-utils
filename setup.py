from setuptools import setup, find_packages

setup(
    name="ai-utils",
    version="0.1.2",
    description="Utility packages for ML, NLP, LLMs",
    author="Mirellys Arteta Davila",
    author_email="mirellys710@gmail.com",
    url="https://github.com/syllerim/ai-utils",
    packages=find_packages(),
    install_requires=[
        "requests",
        "pandas"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
