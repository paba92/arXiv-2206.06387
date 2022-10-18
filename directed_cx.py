
from typing import Union

import numpy as np


def default_h_layout(n: int) -> np.ndarray:
    # Prepare default Hadamard layout (Eq. 31)
    t_h = np.eye(n, dtype=np.uint8)
    t_h[0, 0] = 0
    t_h[1:, 0] = np.ones((n-1, 1), dtype=np.uint8)
    return t_h


def algorithm_1(n: int, t_cz: np.ndarray, return_list: bool = False) -> Union[np.ndarray, list[int]]:
    # Assume T_H in default initial layout (as created by `default_h_layout(n)`)
    # T_H does not have to be stored as a matrix, we can just store the position the H in each row has been moved to
    h_cols = []
    # -------------------------------------------
    # --- Algorithm 1 (Moving Hadamard gates) ---
    # -------------------------------------------
    h_max = 0  # Position of the rightmost H that has already been moved left
    for i in range(1, n):
        try:
            c = np.flatnonzero(t_cz[i, :i])[-1] + 1  # Find position directly after first CZ to the left
        except IndexError:
            # No CZ found
            h_cols.append(0)  # Cancel H in the first layer
            continue
        if c == i:
            # Unable to move left, attempt to move right
            if len(np.flatnonzero(t_cz[i, i:])) == 0:
                # If no CZ is to the right, move $\Gate{H}$ to the last layer, ...
                h_cols.append(n-1)
            else:
                # ...otherwise remain in place.
                h_cols.append(i)
                h_max = i
        else:
            h_max = max(h_max, c)  # Find the more restrictive condition (either CZ on current qubit or H on previous)
            h_cols.append(h_max)  # Move H to the target layer

    if return_list:
        # Directly return the list constructed by the loop above
        return h_cols
    else:
        # Build T_H as in the paper
        t_h = np.zeros((n, n), dtype=np.uint8)
        for im1, j in enumerate(h_cols):
            if j == 0:
                pass  # Cancellation has taken place, remains an all-zero row
            else:
                t_h[im1+1, [0, j]] = 1, 1  # Place H in the first and j-th row
        return t_h
