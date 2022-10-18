
from typing import Union

import numpy as np

from qiskit import QuantumCircuit


def default_h_layout(n: int) -> np.ndarray:
    # Prepare default Hadamard layout (Eq. 31)
    t_h = np.eye(n, dtype=np.uint8)
    t_h[0, 0] = 0
    t_h[1:, 0] = np.ones(n-1, dtype=np.uint8)
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
    # RETURN
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


def algorithm_2(n: int, t_h: np.ndarray, t_cz: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    ## Algorithm 2
    #
    # Generate the CZ/GCZ
    seq = []
    cur_gcz = []
    for i in range(n-1):
        if sum(t_h[:, i]) > 0:  # There is at least one hadamard gate at column i
            if sum(t_h[:, i+1]) > 0:  # If the ith column is surrounded by hadamard layers.
                if cur_gcz:  # If CZ is non empty, then append the next column of t_cz to CZ
                    if sum(t_cz[:, i]) > 1:
                        cur_gcz.append(t_cz[:, i])
                    seq.append(cur_gcz)  # Add CZ to the sequence of global CZ gates
                    cur_gcz = []
                else:  # CZ is empty
                    twoq_cz = np.copy(t_h[:, i+1])
                    twoq_cz[i] = 1
                    seq.append(twoq_cz.reshape((n, 1)))  # Append a two qubit CZ (first part of the column of t_cz)
                    remainder_cz = (t_cz[:, i] + t_h[:, i+1]) % 2  # The second part of the column of t_cz
                    if sum(remainder_cz) < 2:
                        cur_gcz = []
                    else:
                        cur_gcz = [remainder_cz]
            else:
                if sum(t_cz[:, i]) > 1:
                    cur_gcz.append(t_cz[:, i])
                seq.append(cur_gcz)  # TODO: This may append empty GCZ layers to the sequence!
                cur_gcz = []
        else:
            if sum(t_cz[:, i]) > 1:
                seq[-1].append(t_cz[:, i])
    # RETURN
    return (
        np.delete(t_h, np.argwhere(np.all(t_h[..., :] == 0, axis=0)), axis=1),  # truncated H pattern
        [(np.column_stack(lay)  # assemble list of fan-out-like gates to GCZ layer matrix
          if isinstance(lay, list) else
          lay)  # two-qubit CZs are already in the desired format
         for lay in seq]  # sequence of (two-qubit) CZ and GCZ layers
    )


# def directed_cx_to_h_gcz(n: int, b: np.ndarray) -> list[list[Union[int, tuple[int, int]]]]:
#     pass


def create_random_directed_cx(n: int):
    t_cz = np.tril(np.random.randint(0, 2, (n, n), np.uint8), -1)  # populate strictly lower triangle at random
    for j in range(n-1):
        if sum(t_cz[j+1:, j]):
            t_cz[j, j] = 1  # set control on diagonal, if any target has been randomly selected in that column
    return t_cz


def to_qiskit(n: int, t_h, seq):
    assert t_h.shape[1] == len(seq) + 1  # TODO: Can this fail? When?
    qc = QuantumCircuit(n)
    for h_lay, cz_lay in zip(t_h.T, seq):
        for i in np.flatnonzero(h_lay):
            qc.h(i)
        qc.barrier()
        for fanout in cz_lay.T:
            idcs = np.flatnonzero(fanout)
            try:
                ctrl = idcs[0]
            except IndexError:
                continue  # skips empty layers. should only ever happen when called from `to_qiskit_prealg()`
            for trgt in idcs[1:]:
                qc.cz(ctrl, trgt)
        qc.barrier()
    for i in np.flatnonzero(t_h[:, -1]):
        qc.h(i)
    return qc


def to_qiskit_prealg(n: int, t_cz):
    t_h = default_h_layout(n)
    seq = list(lay.reshape((n, 1)) for lay in t_cz.T)[:-1]
    return to_qiskit(n, t_h, seq)


if __name__ == '__main__':
    n = 5

    t_cz = create_random_directed_cx(n)
    # print("Random directed CX layer layout:")
    # print(t_cz)
    print(to_qiskit_prealg(n, t_cz))

    t_h = algorithm_1(n, t_cz)
    # print("Corresponding modified H pattern (post-Alg1):")
    # print(t_h)
    # print()

    red_t_h, seq = algorithm_2(n, t_h, t_cz)
    # print("Reduced H pattern after GCZ formation (post-Alg2):")
    # print(red_t_h)
    # print("Diagonal gate sequence:")
    # for i, lay in enumerate(seq):
    #     print(f"-> layer {i}:")
    #     print(lay)

    print(to_qiskit(n, red_t_h, seq))
