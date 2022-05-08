# Author: Nathan Trouvain at 07/05/2022 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from collections import namedtuple
from pathlib import Path

DATA_FOLDER = Path.home() / Path("reservoirpy-data")


def _get_data_folder(folder_path=None):
    if folder_path is None:
        folder_path = DATA_FOLDER
    else:
        folder_path = Path(folder_path)

    if not folder_path.exists():
        folder_path.mkdir(parents=True)

    return folder_path
