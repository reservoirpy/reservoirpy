# Author: Nathan Trouvain at 08/03/2022 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from .intrinsic_plasticity import IPReservoir
from .nvar import NVAR
from .reservoir import Reservoir

__all__ = ["Reservoir", "IPReservoir", "NVAR"]
