"""This module provides auxiliary functions for locpoly."""
import numpy as np
import math

from numba import jit


@jit(nopython=True)
def get_kernel_weights(tau, bandwidth, delta):
    """This function computes approximated weights for the Gaussian kernel."""
    L = math.floor(tau * bandwidth / delta)
    lenfkap = 2 * L + 1
    fkap = np.zeros(lenfkap)

    # Determine midpoint of fkap
    mid = L + 1

    # Compute the kernel weights
    for j in range(L + 1):

        # Note that the mid point (fkap[mid - 1]) receives a weight of 1.
        fkap[mid - 1 + j] = math.exp(-(delta * j / bandwidth) ** 2 / 2)

        # Because of the kernel's symmetry, weights in equidistance
        # below and above the midpoint are identical.
        fkap[mid - 1 - j] = fkap[mid - 1 + j]

    return L, lenfkap, fkap, mid


@jit(nopython=True)
def combine_bincounts_kernel_weights(
    xcounts, ycounts, M, dimss, dimtt, L, lenfkap, fkap, mid, delta
):
    """
    This function combines the bin counts (xcounts) and bin averages (ycounts) with
    kernel weights via a series of direct convolutions. As a result, binned
    approximations to X'W X and X'W y, denoted by ss and tt, are computed.

    Recall that the local polynomial curve estimator beta_ and its derivatives are
    minimizers to a locally weighted least-squares problem. At each grid
    point g = 1,..., M in the grid, beta_ is computed as the solution to the
    linear matrix equation:

    X'W X * beta_ = X'W y,

    where W are kernel weights approximated by the Gaussian density function.
    X'W X and X'W y are approximated by ss and tt,
    which are the result of a direct convolution of bin counts (xcounts) and kernel
    weights, and bin averages (ycounts) and kernel weights, respectively.

    The terms "kernel" and "kernel function" are used interchangeably
    throughout.

    For more information see the documentation of the main function locpoly
    under KernReg.locpoly.

    Parameters
    ----------
    xcounts: np.ndarry
        1-D array of binned x-values ("bin counts") of length M.
    ycounts: np.ndarry
        1-D array of binned y-values ("bin averages") of length M.
    M: int
        Gridsize, i.e. number of equally-spaced grid points.
    dimss: int
        Number of columns of output array ss, i.e. the binned approximation to X'W X.
    dimtt: int
        Number of columns of output array tt, i.e the binned approximation to X'W y.
    lenfkap: int
        Length of 1-D array fkap.
    fkap: np.ndarry
        1-D array of length lenfkap containing
        approximated weights for the Gaussian kernel
        (W in the notation above).
    L: int
        Parameter defining the number of times the kernel function
        has to be evaluated.
        Note that L < N, where N is the total number of observations.
    mid: int
        Midpoint of fkap.
    delta: float
        Bin width.

    Returns
    -------
    ss: np.ndarry
        Dimensions (M, ppp). Binned approximation to X'W X.
    tt: np.ndarry
        Dimensions (M, pp). Binned approximation to X'W y.
    """
    ss = np.zeros((M, dimss))
    tt = np.zeros((M, dimtt))

    for g in range(M):
        if xcounts[g] != 0:
            for i in range(max(0, g - L - 1), min(M, g + L)):

                if 0 <= i <= M - 1 and 0 <= g - i + mid - 1 <= lenfkap - 1:
                    fac = 1

                    ss[i, 0] += xcounts[g] * fkap[g - i + mid - 1]
                    tt[i, 0] += ycounts[g] * fkap[g - i + mid - 1]

                    for j in range(1, dimss):
                        fac = fac * delta * (g - i)

                        ss[i, j] += xcounts[g] * fkap[g - i + mid - 1] * fac

                        if j < dimtt:
                            tt[i, j] += ycounts[g] * fkap[g - i + mid - 1] * fac
    return ss, tt


@jit(nopython=True)
def get_curve_estimator(ss, tt, dimtt, derivative, M):
    """
    This functions solves the locally weighted least-squares regression
    problem and returns an estimator for the v-th derivative of beta_,
    the local polynomial estimator, at all points in the grid.
    """
    Smat = np.zeros((dimtt, dimtt))
    Tvec = np.zeros(dimtt)
    curvest = np.zeros(M)

    for g in range(M):
        for i in range(0, dimtt):
            for j in range(0, dimtt):
                indss = i + j
                Smat[i, j] = ss[g, indss]

                Tvec[i] = tt[g, i]

        # Calculate beta_ as the solution to the linear matrix equation
        # X'W X * beta_ = X'W y.
        # Note that Smat and Tvec are binned approximations to X'W X and
        # X'W y, respectively, evaluated at the given grid point.
        beta_ = np.linalg.solve(Smat, Tvec)

        # Obtain curve estimator for the desired derivative of beta_.
        curvest[g] = beta_[derivative]

    curvest = math.gamma(derivative + 1) * curvest

    return curvest


@jit
def is_sorted(a):
    """This function checks if the input array is sorted ascendingly."""
    for i in range(a.size - 1):
        if a[i + 1] < a[i]:
            return False
        return True
