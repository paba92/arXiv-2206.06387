
import itertools as it
from typing import Union

import numpy as np
import cvxpy as cp


# ======================
# === UTILITY CACHES ===
# ======================

tril_indices = {}
outer_prods = {}


def get_tril_indices(n: int, *, save_in_cache: bool = True) -> tuple[np.ndarray, np.ndarray]:
    try:
        return tril_indices[n]
    except KeyError:
        res = np.tril_indices(n, -1)
        if save_in_cache:
            tril_indices[n] = res
        return res


def get_outer_prods(n: int, *, save_in_cache: bool = True) -> np.ndarray:
    try:
        return outer_prods[n]
    except KeyError:
        ms = [(1,) + tail for tail in it.product([1, -1], repeat=(n-1))]
        i_lower = get_tril_indices(n, save_in_cache=save_in_cache)
        res = np.column_stack([np.outer(m, m)[i_lower] for m in ms])
        if save_in_cache:
            outer_prods[n] = res
        return res


# =================================
# === ENCODING SEQUENCE SOLVERS ===
# =================================

def lp_approach(n: int, m: np.ndarray, *, save_in_cache: bool = True, threshold: Union[float, None] = 1e-12) \
        -> np.ndarray:
    # check input:
    if m.shape == (n, n):  # input is square matrix...
        y = m[get_tril_indices(n, save_in_cache=save_in_cache)]  # ... and lower triangle is extracted
    elif m.shape == (n*(n-1) // 2,):  # input is already vectorized
        y = m
    else:
        raise ValueError(f"Invalid input shape {m.shape}")
    # build LP as CVXPY model
    x = cp.Variable(1 << (n-1), nonneg=True)  # non-negativity constraint
    objective = cp.Minimize(cp.norm(x, 1))  # L1 objective function
    constraints = [get_outer_prods(n, save_in_cache=save_in_cache) @ x == y]  # linear equation system
    prob = cp.Problem(objective, constraints)
    # solve LP using simplex method
    prob.solve(solver=cp.GLPK, glpk={'msg_lev': 'GLP_MSG_OFF'})
    # extract solution and potentially suppress negligibly small entries
    solution = x.value
    if threshold is not None:
        solution[solution < threshold] = 0.0
    # return result
    return solution


def mip_approach(n: int, m: np.ndarray, alpha: float, c_l: float, c_u_scaling: float, rel_opt_tol: float,
                 *, save_in_cache: bool = True, threshold: Union[float, None] = 1e-12) -> np.ndarray:
    # check input:
    if m.shape == (n, n):  # input is square matrix...
        y = m[get_tril_indices(n, save_in_cache=save_in_cache)]  # ... and lower triangle is extracted
    elif m.shape == (n*(n-1) // 2,):  # input is already vectorized
        y = m
    else:
        raise ValueError(f"Invalid input shape {m.shape}")
    # build MIP as CVXPY model
    n2 = 1 << (n-1)
    z = cp.Variable(n2, boolean=True)  # integer variables (either 0 or 1)
    t = cp.Variable(n2, nonneg=True)  # non-negative continuous variables
    # simplify extreme cases of the objective function
    if alpha <= 0.:
        objective = cp.Minimize(cp.norm(z, 1))
    elif alpha >= 1.:
        objective = cp.Minimize(cp.norm(t, 1))
    else:
        objective = cp.Minimize(alpha * cp.norm(t, 1) + (1.-alpha) * cp.norm(z, 1))
    c_u = c_u_scaling * np.max(y)  # compute adaptive upper bound
    constraints = [cp.multiply(c_l, z) <= t, t <= cp.multiply(c_u, z)]  # limit constraints
    constraints.append(get_outer_prods(n, save_in_cache=save_in_cache) @ t == y)  # linear equation system
    prob = cp.Problem(objective, constraints)
    # solve MIP using MOSEK
    prob.solve(solver=cp.MOSEK, verbose=False, mosek_params={"MSK_DPAR_MIO_TOL_REL_GAP": rel_opt_tol})
    # extract solution and potentially suppress negligibly small entries
    solution = t.value
    if threshold is not None:
        solution[solution < threshold] = 0.0
    # return result
    return solution
