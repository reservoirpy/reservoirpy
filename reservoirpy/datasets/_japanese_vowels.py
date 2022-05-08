"""Japanese vowels dataset."""
# Author: Nathan Trouvain at 07/05/2022 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from io import StringIO
from pathlib import Path
from urllib.request import urlopen

import joblib
import numpy as np

from .. import logger
from ._utils import _get_data_folder

SOURCE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/JapaneseVowels-mld/"
)

REMOTE_FILES = {
    "DESCR": "JapaneseVowels.data.html",
    "train": "ae.train",
    "test": "ae.test",
    "train_sizes": "size_ae.train",
    "test_sizes": "size_ae.test",
}

# class labels
SPEAKERS = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]

ONE_HOT_SPEAKERS = np.eye(9)


def _format_data(data, block_numbers, one_hot_encode):
    """Load and parse data from downloaded binary files."""

    X = []
    Y = []

    data = data.decode("utf-8").split("\n\n")[:-1]

    block_cursor = 0
    speaker_cursor = 0
    for block in data:

        if block_cursor >= block_numbers[speaker_cursor]:
            block_cursor = 0
            speaker_cursor += 1

        X.append(np.loadtxt(StringIO(block)))

        if one_hot_encode:
            Y.append(ONE_HOT_SPEAKERS[speaker_cursor].reshape(1, -1))
        else:
            Y.append(np.array([SPEAKERS[speaker_cursor]]).reshape(1, 1))

        block_cursor += 1

    return X, Y


def _download(data_folder, file_name, file_role):  # pragma: no cover
    """Download data from source into the reservoirpy data local directory."""

    logger.info(f"Downloading {SOURCE_URL + file_name}.")

    file_path = Path(file_name)

    with urlopen(SOURCE_URL + file_name) as f:

        # extract data block sizes integer lists
        if file_role in ["train_sizes", "test_sizes"]:
            data = [s for s in f.read().decode("utf-8").split(" ")]

            # remove empty characters and spaces
            data = [int(s) for s in filter(lambda s: s not in ["", "\n", " "], data)]

        else:
            data = f.read()

        joblib.dump(data, data_folder / file_path, compress=6)

    return data


def _repeat_target(blocks, targets):
    """Repeat target label/vector along block's time axis."""

    repeated_targets = []
    for block, target in zip(blocks, targets):
        timesteps = block.shape[0]
        target_series = np.repeat(target, timesteps, axis=0)
        repeated_targets.append(target_series)

    return repeated_targets


def japanese_vowels(
    one_hot_encode=True, repeat_targets=False, data_folder=None, reload=False
):
    """Load the Japanese vowels [16]_ dataset.

    This is a classic audio discimination task. Nine male Japanese speakers
    pronounced the ` \\ae\\ ` vowel. The task consists in infering the speaker
    identity from the audio recording.

    Audio recordings are series of 12 LPC cepstrum coefficient. Series contains
    between 7 and 29 timesteps. Each series (or "block") is one utterance of ` \\ae\\ `
    vowel from one speaker.

    ============================   ===============================
    Classes                                                      9
    Samples per class (training)       30 series of 7-29 timesteps
    Samples per class (testing)     29-50 series of 7-29 timesteps
    Samples total                                              640
    Dimensionality                                              12
    Features                                                  real
    ============================   ===============================

    Data is downloaded from:
    https://archive.ics.uci.edu/ml/machine-learning-databases/JapaneseVowels-mld/

    Parameters
    ----------
    one_hot_encode : bool, default to True
        If True, returns class label as a one-hot encoded vector.
    repeat_targets : bool, default to False
        If True, repeat the target label or vector along the time axis of the
        corresponding sample.
    data_folder : str or Path-like object, optional
        Local destination of the downloaded data.
    reload : bool, default to False
        If True, re-download data from remote repository. Else, if a cached version
        of the dataset exists, use the cached dataset.

    Returns
    -------
    X_train, Y_train, X_test, Y_test
        Lists of arrays of shape (timesteps, features) or (timesteps, target)
        or (target,).

    References
    ----------
    .. [16] M. Kudo, J. Toyama and M. Shimbo. (1999).
           "Multidimensional Curve Classification Using Passing-Through Regions".
           Pattern Recognition Letters, Vol. 20, No. 11--13, pages 1103--1111.

    """

    data_folder = _get_data_folder(data_folder)

    data_files = {}
    for file_role, file_name in REMOTE_FILES.items():
        file_path = Path(file_name)

        if not (data_folder / file_path).exists() or reload:  # pragma: no cover
            data_files[file_role] = _download(data_folder, file_name, file_role)
        else:
            data_files[file_role] = joblib.load(data_folder / file_path)

    X_train, Y_train = _format_data(
        data_files["train"], data_files["train_sizes"], one_hot_encode
    )

    X_test, Y_test = _format_data(
        data_files["test"], data_files["test_sizes"], one_hot_encode
    )

    if repeat_targets:
        Y_train = _repeat_target(X_train, Y_train)
        Y_test = _repeat_target(X_test, Y_test)

    return X_train, Y_train, X_test, Y_test
