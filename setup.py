import os

from setuptools import setup, find_packages

NAME = "reservoirpy"

__version__ = None
version_file = os.path.join("reservoirpy", "_version.py")
with open(version_file) as f:
    exec(f.read())

AUTHOR = "Xavier Hinaut"
AUTHOR_EMAIL = "xavier.hinaut@inria.fr"

MAINTAINERS = "Xavier Hinaut, Nathan Trouvain"
MAINTAINERS_EMAIL = "xavier.hinaut@inria.fr, nathan.trouvain@inria.fr"

DESCRIPTION = "A simple and flexible code for Reservoir " \
               "Computing architectures like Echo State Networks."

with open("README.md", "r", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

URL = "https://github.com/reservoirpy/reservoirpy"
DOWNLOAD_URL = (f"{URL}/v{__version__}.tar.gz")

INSTALL_REQUIRES = [
    "tqdm>=4.43.0",
    "joblib>=0.12",
    "dill>=0.3.0"
    'numpy>=1.15.0',
    'scipy>=1.0.0,<=1.7.3',
    'joblib>=0.12',
]

EXTRA_REQUIRES = {
    'hyper': ['matplotlib>=2.2.0', 'hyperopt', 'seaborn'],
}

PROJECT_URLS = {
    'Bug Tracker': 'https://github.com/reservoirpy/reservoirpy/issues',
    'Documentation': 'https://reservoirpy.readthedocs.io/en/latest/index.html',
    'Source Code': URL
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
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering',
            ('Topic :: Scientific/Engineering :: '
             'Artificial Intelligence'),
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX',
            'Operating System :: Unix',
            'Operating System :: MacOS',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            ('Programming Language :: Python :: '
             'Implementation :: PyPy')
        ],
        python_requires='>=3.6',
        install_requires=INSTALL_REQUIRES,
        extra_require=EXTRA_REQUIRES,
    )
