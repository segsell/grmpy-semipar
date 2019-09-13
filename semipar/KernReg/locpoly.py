"""
This module provides a function for the estimation of a local polynomial
kernel regression.
"""
import numpy as np
import math

from semipar.KernReg.locpoly_linbin import bin_counts_and_averages

from semipar.KernReg.locpoly_auxiliary import combine_bincounts_kernel_weights
from semipar.KernReg.locpoly_auxiliary import discretize_bandwidths
from semipar.KernReg.locpoly_auxiliary import get_curve_estimator
from semipar.KernReg.locpoly_auxiliary import get_kernel_weights
from semipar.KernReg.locpoly_auxiliary import is_sorted


def locpoly(
    x,
    y,
    derivative,
    degree,
    bandwidth,
    gridsize=401,
    a=None,
    b=None,
    bwdisc=25,
    binned=False,
    truncate=True,
):
    """
    This function fits a regression function or their derivatives via
    local polynomials. A local polynomial fit requires a weighted
    least-squares regression at every point g = 1,..., M in the grid.
    The Gaussian density function is used as kernel weight.

    It is recommended that for a v-th derivative the order of the polynomial
    be p = v + 1.

    The local polynomial curve estimator beta_ and its derivatives are
    minimizers to the locally weighted least-squares problem. At each grid
    point, beta_ is computed as the solution to the linear matrix equation:

    X'W X * beta_ = X'W y,

    where W are kernel weights. X'W X and X'W y are aproximated by ss and tt,
    which are the result of a direct convolution of bin counts and kernel
    weights, or bin averages and kernel weights, respectively.

    A binned approximation over an equally-spaced grid is used for fast
    computation. Fan and Marron (1994) recommend a default gridsize of M = 400
    for the popular case of graphical analysis. They find that fewer than 400
    grid points results in distracting "granularity", while more grid points
    often give negligible improvements in resolution. Instead of a scalar
    bandwidth, local bandwidths of length gridsize may be chosen.

    The terms "kernel" and "kernel function" are used interchangeably
    throughout and denoted by K.

    This function builds on the R function "locpoly" from the "KernSmooth"
    package maintained by Brian Ripley and the original Fortran routines
    provided by M.P. Wand.

    Parameters
    ----------
    x: np.ndarry
        Array of x data. Missing values are not accepted. Must be sorted.
    y: np.ndarry
        1-D Array of y data. This must be same length as x. Missing values are
        not accepted. Must be presorted by x.
    derivative: int
        Order of the derivative to be estimated.
    degree: int:
        Degree of local polynomial used. Its value must be greater than or
        equal to the value of drv. Generally, users should choose a degree of
        size drv + 1.
    bandwidth: int, float, list or np.ndarry
        Kernel bandwidth smoothing parameter. It may be a scalar or a array of
        length gridsize.
    gridsize: int
        Number of equally-spaced grid points over which the function is to be
        estimated.
    a: float
        Start point of the grid mesh.
    b: float
        End point of the grid mesh.
    bwdisc: int
        Number of logarithmically equally-spaced bandwidths on which local
        bandwidths are discretized, to speed up computation.
        Only relevant if bandwidth is a list or np.ndarry of length gridsize.
    binned: bool
        If True, then x and y are taken to be bin counts rather than raw data
        and the binning step is skipped.
    truncate: bool
        If True, then endpoints are truncated.

    Returns
    -------
    gridpoints: np.ndarry
        Array of sorted x values, i.e. grid points, at which the estimate
        of E[Y|X] (or its derivative) is computed.
    curvest: np.ndarry
        Array of M local estimators.
    """
    # The input arrays x and y, i.e. predictor and response variable,
    # respectively, must be sorted by x.
    if is_sorted(x) is False:
        raise Warning("Input arrays x and y must be sorted by x " "before estimation!")

    if a is None:
        a = min(x)

    if b is None:
        b = max(x)

    # Rename global variables
    drv = derivative
    bw = bandwidth
    M = gridsize

    pp = degree + 1
    ppp = 2 * degree + 1

    # tau is chosen so that the interval [-tau, tau] is the
    # "effective support" of the Gaussian kernel,
    # i.e. K is effectively zero outside of [-tau, tau].
    # According to Wand (1994) and Wand & Jones (1995) tau = 4 is a
    # reasonable choice for the Gaussian kernel.
    tau = 4

    # Set the bin width
    delta = (b - a) / (M - 1)

    # 1a. Bin the data if not already binned
    if binned is False:
        xcounts, ycounts = bin_counts_and_averages(x, y, M, a, delta, truncate)
    else:
        xcounts, ycounts = x, y

    # 1b. Discretize the bandwidths (only necessary if len(bw) == M)
    if isinstance(bw, (float, int)) or len(bw) == 1:
        Q = 1
        L = math.floor(tau * bw / delta)
        hdisc = bw
        # Index set
        indic = np.ones(M)

    else:
        Q = bwdisc
        hdisc, L, indic = discretize_bandwidths(bw, M, Q, delta)

    # 2. Obtain kernel weights
    # Note that only L < N kernel evaluations are required to obtain the
    # kernel weights, fkap, regardless of the number of observations N.
    fkap, dimfkap, midpt = get_kernel_weights(L, Q, delta, hdisc)

    # 3. Combine bin counts and kernel weights
    ss, tt = combine_bincounts_kernel_weights(
        xcounts, ycounts, Q, M, pp, ppp, dimfkap, fkap, L, midpt, indic, delta
    )

    # Fit the curve and obtain estimator for the desired derivative
    curvest = get_curve_estimator(ss, tt, pp, drv, M)
    curvest = math.gamma(drv + 1) * curvest

    # Generate grid points for visual representation
    gridpoints = np.linspace(a, b, M)

    return gridpoints, curvest
