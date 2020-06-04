import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="reservoirpy",
    version="0.2.0",
    author="Xavier Hinaut",
    author_email="xavier.hinaut@inria.fr",
    description="A simple and flexible code for Reservoir Computing architectures like Echo State Networks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/neuronalX/reservoirpy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    python_requires='>=3.6',
    install_requires=[
        "tqdm>=4.43.0",
        "joblib>=0.12",
        "dill>=0.3.0"
        'numpy>=1.15.0',
        'scipy>=1.0.0',
        'joblib>=0.12',
    ]
)
