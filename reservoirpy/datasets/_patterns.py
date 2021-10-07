# Author: Nathan Trouvain at 04/10/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import string

import numpy as np

from _seed import get_seed


def asabuki_1alphachunks(n_timesteps,
                         chunk=('a', 'b', 'c', 'd'),
                         width=50,
                         input_gain=2,
                         min_length=5,
                         dt=1):

    alphabet = tuple(string.ascii_lowercase)
    random_elements = tuple(set(alphabet) - set(chunk))

    input_shape = np.zeros(width * 2)

    input_shape[0:width] = input_gain * (1 - np.exp(-(np.arange(0, width) / 10)))
    input_shape[width:2 * width] = input_gain * np.exp(-(np.arange(0, width) / 10))


    simtime = np.arange(0, n_timesteps * dt, dt)
    simtime_len = len(simtime)

    symbol_list = np.zeros(simtime_len)
    target_list = np.zeros(simtime_len)

    random_seq_len = np.random.randint(min_length, min_length + 4)
    random_seq = [''] * random_seq_len
    for i in range(random_seq_len):
        random_seq[i] = random_elements[
            np.random.randint(0, len(random_elements))]
    input_type = random_seq
    m = 0

    I = np.zeros((len(alphabet), simtime_len))
    for i in range(simtime_len):
        if (i % width == 0 and i > 0):
            if input_type == chunk:
                if m == len(chunk) - 1:
                    random_seq_len = np.random.randint(min_length,
                                                       min_length + 4)
                    random_seq = [''] * random_seq_len
                    for l in range(random_seq_len):
                        random_seq[l] = random_elements[
                            np.random.randint(0, len(random_elements))]

                    input_type = random_seq
                    m = 0

                else:
                    input_type = chunk
                    m += 1

            elif input_type == random_seq:
                if m == len(random_seq) - 1:
                    input_type = chunk
                    m = 0

                else:
                    input_type = random_seq
                    m += 1

            I[alphabet.index(input_type[m]),
            i:min(i + width * 2, simtime_len)] = input_shape[
                                                 0:min(i + width * 2,
                                                       simtime_len) - i]
