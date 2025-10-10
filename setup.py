import os

from setuptools import find_packages, setup

NAME = "reservoirpy"

__version__ = ""
version_file = os.path.join("reservoirpy", "_version.py")
with open(version_file) as f:
    exec(f.read())

AUTHOR = "Xavier Hinaut"
AUTHOR_EMAIL = "xavier.hinaut@inria.fr"

MAINTAINERS = "Xavier Hinaut, Paul Bernard"
MAINTAINERS_EMAIL = "xavier.hinaut@inria.fr, paul.bernard@inria.fr"

DESCRIPTION = "A simple and flexible code for Reservoir Computing architectures like Echo State Networks."

with open("README.md", "r", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

URL = "https://github.com/reservoirpy/reservoirpy"
DOWNLOAD_URL = f"{URL}/v{__version__}.tar.gz"

INSTALL_REQUIRES = [
    "joblib>=0.14.1",
    "numpy>=1.21.1",
    "scipy>=1.4.1",
]

EXTRA_REQUIRES = {
    "hyper": ["hyperopt", "matplotlib>=2.2.0", "tqdm>=4.43.0"],
    "sklearn": ["scikit-learn (>=0.24.2, <1.7.0)"],
    "jax": ["jax>=0.4"],
}

PROJECT_URLS = {
    "Bug Tracker": "https://github.com/reservoirpy/reservoirpy/issues",
    "Documentation": "https://reservoirpy.readthedocs.io/en/latest/index.html",
    "Source Code": URL,
    "Release notes": "https://github.com/reservoirpy/reservoirpy/releases",
}

if __name__ == "__main__":
    setup(
        name=NAME,
        version=__version__,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        maintainer=MAINTAINERS,
        maintainer_email=MAINTAINERS_EMAIL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        url=URL,
        project_urls=PROJECT_URLS,
        download_url=DOWNLOAD_URL,
        packages=find_packages(),
        classifiers=[
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX",
            "Operating System :: Unix",
            "Operating System :: MacOS",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Programming Language :: Python :: 3.13",
            "Programming Language :: Python :: Implementation :: PyPy",
        ],
        python_requires=">=3.9",
        install_requires=INSTALL_REQUIRES,
        extras_require=EXTRA_REQUIRES | {"all": sum(EXTRA_REQUIRES.values(), [])},
        package_data={"reservoirpy": ["datasets/santafe_laser.npy"]},
    )
