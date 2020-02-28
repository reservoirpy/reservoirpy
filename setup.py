import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="reservoirpy", # Replace with your own username
    version="0.0.1",
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
)
