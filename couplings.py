
import numpy as np
from numpy import linalg
from scipy.optimize import root

import matplotlib.pyplot as plt

import warnings
import itertools


class TotalPotential1D:
    """A potential acting on `n` particles that live in one-dimensional space."""

    def __init__(self, n, **config):
        self._n = n

    @property
    def n(self):
        return self._n

    def potential(self, z):
        raise NotImplementedError()

    def dk_tensor(self, z, k):
        raise NotImplementedError()

    def gradient(self, z):
        return self.dk_tensor(z, 1)

    def hessian(self, z):
        return self.dk_tensor(z, 2)


class LocalPotential1D:
    """A potential acting on a single particle that lives in one-dimensional space."""

    def potential(self, z):
        raise NotImplementedError()

    def derivative(self, z, order=1):
        if order > 0:
            from scipy.misc import derivative
            return derivative(self.potential, z, n=order)
        elif order == 0:
            return self.potential(z)
        else:
            raise ValueError(f"`order` must be a non-negative integer, not: {order}")


class TotalFromLocal(TotalPotential1D):
    """The total potential on `n` particles caused by a given local potential on each individual particle."""

    def __init__(self, n: int, local: LocalPotential1D, **config):
        super().__init__(n, **config)
        self._local = local  # local potential

    def potential(self, z):
        return sum(self._local.potential(z[i]) for i in range(self._n))

    def dk_tensor(self, z, k):
        return np.fromiter(map(lambda idx: (self._local.derivative(z[idx[0]], k) if (len(set(idx)) == 1) else 0),
                               itertools.product(range(self._n), repeat=k)),
                           np.float_, self._n**k).reshape(k*(self._n,))

    def gradient(self, z):
        # return np.fromfunction(lambda i: self._local.derivative(1, z[i]),
        #                        (self._n,), dtype=np.int)
        return self.dk_tensor(z, 1)

    def hessian(self, z):
        # return np.fromfunction(lambda i, j: (self._local.derivative(2, z[i]) if i == j else 0),
        #                        (self._n, self._n), dtype=np.int)
        return self.dk_tensor(z, 2)


class PairwiseCoulomb1D(TotalPotential1D):
    """The mutual Coulomb potential of `n` particles in one spatial dimension."""

    def __init__(self, n: int, q=1, e2_4pieps0=1, **config):
        super().__init__(n, **config)
        self._q2_4pieps0 = q**2 * e2_4pieps0  # q^2/(4*pi*eps0)

    @staticmethod
    def _potential_raw_function(n, z):
        return sum(1 / abs(z[i] - z[j]) for i in range(n) for j in range(i))

    @staticmethod
    def _gradient_raw_function(n, z):
        def gradient_entry(i):
            return (sum(1 / (z[i] - z[j]) ** 2 for j in range(i+1, n))
                    - sum(1 / (z[i] - z[j]) ** 2 for j in range(i)))
        return gradient_entry

    @staticmethod
    def _hessian_raw_function(n, z):
        def hessian_entry(ij):
            i, j = ij
            if i == j:
                return sum(1 / abs(z[i] - z[k]) ** 3 for k in range(n) if k != i)
            else:
                return -1 / abs(z[i] - z[j]) ** 3
        return hessian_entry

    @staticmethod
    def _d3_tensor_raw_function(n, z):
        def d3_two_sites(i, k):
            if i > k:
                return 1 / (z[i] - z[k]) ** 4
            else:
                return -1 / (z[i] - z[k]) ** 4
        def d3_tensor_entry(ijk):
            i, j, k = ijk
            if i == j == k:
                return (sum(1 / (z[i] - z[l]) ** 4 for l in range(i+1, n))
                        - sum(1 / (z[i] - z[l]) ** 4 for l in range(i)))
            elif i == j:
                return d3_two_sites(i, k)
            elif j == k:
                return d3_two_sites(j, i)
            elif k == i:
                return d3_two_sites(k, j)
            else:
                return 0
        return d3_tensor_entry

    def potential(self, z):
        return self._q2_4pieps0 * self._potential_raw_function(self._n, z)

    def gradient(self, z):
        # return self._q2_4pieps0 * np.fromfunction(self._gradient_raw_function(self._n, z),
        #                                           (self._n,), dtype=np.int)
        return self._q2_4pieps0 * np.fromiter(map(self._gradient_raw_function(self._n, z),
                                                  range(self._n)),
                                              np.float_, self._n)

    def hessian(self, z):
        # return 2*self._q2_4pieps0 * np.fromfunction(self._hessian_raw_function(self._n, z),
        #                                             (self._n, self._n), dtype=np.int)
        return 2*self._q2_4pieps0 * np.fromiter(map(self._hessian_raw_function(self._n, z),
                                                    itertools.product(range(self._n), repeat=2)),
                                                np.float_, self._n**2).reshape(2*(self._n,))

    def dk_tensor(self, z, k):
        if k == 0:
            return self.potential(z)
        elif k == 1:
            return self.gradient(z)
        elif k == 2:
            return self.hessian(z)
        elif k == 3:
            # return 6*self._q2_4pieps0 * np.fromfunction(self._d3_tensor_raw_function(self._n, z),
            #                                             (self._n, self._n, self._n), dtype=np.int)
            return 6*self._q2_4pieps0 * np.fromiter(map(self._d3_tensor_raw_function(self._n, z),
                                                        itertools.product(range(self._n), repeat=3)),
                                                    np.float_, self._n**3).reshape(3*(self._n,))
        else:
            raise NotImplementedError()


class Quadratic1D(LocalPotential1D):
    """A simple parabolic potential."""

    def __init__(self, mw2=1, **config):
        self._mw2 = mw2  # m*\omega^2

    def potential(self, z):
        return (self._mw2 * z**2) / 2

    def derivative(self, z, order=1):
        if order < 1:
            return super().derivative(z, order)
        elif order == 1:
            return self._mw2 * z
        elif order == 2:
            return self._mw2
        else:
            return 0


def get_equilibrium_positions(n, external, init=None, tries=5):
    if isinstance(n, PairwiseCoulomb1D):
        coulomb = n
        n = coulomb.n
    else:
        coulomb = PairwiseCoulomb1D(n)
    if isinstance(external, LocalPotential1D):
        external = TotalFromLocal(n, external)

    def total_gradient(z):
        return external.gradient(z) + coulomb.gradient(z)

    def total_hessian(z):
        return external.hessian(z) + coulomb.hessian(z)

    # TODO: use non-linear spacing
    if init is None:
        outer = 2.16789415 * pow(n, 0.36617105) - 2.16584614  # approximate position of the ion with the greatest z
        init = np.linspace(-outer, outer, n)

    for i in range(tries):
        res = root(total_gradient, init, jac=total_hessian)
        if res.success:
            return res.x
        else:
            warnings.warn(f"Failed to find equilibrium positions on try {i+1}: {res.message}")
            # TODO: rework (maybe)
            init += 0.2 * (np.random.random_sample(n) - 0.5)
            init.sort()
    raise RuntimeError(f"Giving up to find equilibrium position after {tries} tries.")


def normal_mode_matrix(n, external, init=None, tries=5):
    if isinstance(n, PairwiseCoulomb1D):
        coulomb = n
        n = coulomb.n
    else:
        coulomb = PairwiseCoulomb1D(n)
    if isinstance(external, LocalPotential1D):
        external = TotalFromLocal(n, external)

    def total_hessian(zs):
        return coulomb.hessian(zs) + external.hessian(zs)

    return total_hessian(get_equilibrium_positions(coulomb, external, init, tries))


def coupling_matrix(n, external, init=None, tries=5):
    return linalg.inv(normal_mode_matrix(n, external, init, tries))


# === UTIL ===

def plot_matrix(mat, no_diag=False):
    if no_diag:
        mat = mat.copy()
        for i in range(len(mat)):
            mat[i, i] = 0
    plt.imshow(mat)
    plt.colorbar()
    plt.show()


# === EXAMPLE CODE ===

if __name__ == '__main__':
    n_max = 20
    external = Quadratic1D()
    for n in range(2, n_max+1):
        print(f'=====\nJ({n}) =')
        print(coupling_matrix(n, external))
