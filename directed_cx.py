
from typing import Union

import numpy as np


def default_h_layout(n: int) -> np.ndarray:
    # Prepare default Hadamard layout (Eq. 31)
    t_h = np.zeros((n, n), dtype=np.uint8)
    t_h[1:, 0] = np.ones((n-1, 1), dtype=np.uint8)
    t_h[1:, 1:] = np.eye(n-1, dtype=np.uint8)
    return t_h


def algorithm_1(n: int, t_cz: np.ndarray, return_list: bool = False) -> Union[np.ndarray, list[int]]:
    # Assume T_H in default initial layout (as created by `default_h_layout(n)`)
    # --- Algorithm 1 (Moving Hadamard gates) ---
    h_cols = []  # T_H does not have to be stored as matrix, we can just store the col the H in each row has been moved to
    max_nonzero_h = 0  # This stores the col of the rightmost H that has not been moved to the very front
    for i in range(1, n):
        try:
            c = np.flatnonzero(t_cz[i, :i])[-1] + 1
        except IndexError:
            h_cols.append(0)
            continue
        if c == i:
            if len(np.flatnonzero(t_cz[i, i:])) == 0:
                h_cols.append(n-1)
            else:
                h_cols.append(i)
                max_nonzero_h = i
        else:
            max_nonzero_h = max(max_nonzero_h, c)
            h_cols.append(max_nonzero_h)

    if return_list:
        return h_cols
    else:
        # Build T_H as in the paper
        t_h = np.zeros((n, n), dtype=np.uint8)
        for im1, j in enumerate(h_cols):
            if j != 0:
                t_h[im1+1, [0, j]] = 1, 1
        return t_h
