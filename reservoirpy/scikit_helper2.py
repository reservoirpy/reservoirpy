from functools import wraps
from typing import Callable

import numpy as np
from sklearn import linear_model

def get_linear(name) -> Callable: 
	return getattr(linear_model, name)
