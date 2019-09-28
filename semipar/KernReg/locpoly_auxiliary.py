"""This module provides auxiliary functions for locpoly."""
import numpy as np
import numba
import math


def discretize_bandwidths(bw, M, bwdisc, delta):
    """
    Discretize local bandwidths if a list or np.ndarry of M local bandwidths
    is given.

    Parameters
    ----------
    bw: list or np.ndarry
        List or array of M local bandwiths.
    M: int
        Gridsize: Number of equally-spaced grid points.
    bwdisc: int
        Number of logarithmically equally-spaced bandwidths
        on which local bandwidths are discretized, to speed up computation.
    delta: float
        Bin width.

    Returns
    -------
    hdisc: np.ndarry
        Array of discretized local bandwidths.
    L: np.ndarry
        Array of length 'bwdisc' determining the number of times the kernel function
        has to be evaluated.
        Note that L < N, where N is the total number of observations.
    indic: np.ndarry
        Array of length M containing index values.
    """
    indic = np.ones(M)

    # tau is chosen so that the interval [-tau, tau] is the
    # "effective support" of the Gaussian kernel,
    # i.e. K is effectively zero outside of [-tau, tau].
    # Opposed to kernel functions with bounded support,
    # the Gaussian kernel, which has infinite support, is merely
    # "effectively zero" beyond tau.
    # According to Wand (1994) and Wand & Jones (1995) tau = 4 is a
    # reasonable choice for the Gaussian kernel.
    tau = 4

    if len(bw) == M:
        hlow = min(bw)
        hupp = max(bw)

        hdisc = np.exp(np.linspace(math.log(hlow), math.log(hupp), bwdisc))

        # Determine value of L for each member of hdisc
        L = np.floor(tau * hdisc / delta).astype(int)

        # Determine index of closest entry of hdisc
        # to each member of 'bandwidth'
        if bwdisc > 1:
            log_hdisc = np.log(hdisc)
            gap = (log_hdisc[bwdisc - 1] - log_hdisc[0]) / (bwdisc - 1)
            if gap == 0:
                indic = np.ones(M)
            else:
                indic = np.round(((np.log(bw) - math.log(min(bw))) / gap + 1))

    elif isinstance(bw[0], list):
        raise Warning(
            "Input list is not 1-D / is multidimensional. "
            "Please provide a list of the form bw = [0.1, 0.2, ..]."
        )

    else:
        raise Warning("'bandwidth' must be a scalar or np.ndarray of length 'gridsize'")

    if min(L) == 0:
        raise Warning(
            "Binning grid too coarse for current small bandwidth: "
            "Consider increasing 'gridsize' M!"
        )

    return hdisc, L, indic


def get_kernel_weights(L, bwdisc, delta, hdisc):
    """This function computes approximated weights for the Gaussian kernel."""
    # If bandwidth is a scalar, midpt is an integer
    if bwdisc == 1:
        dimfkap = 2 * L + 1
        fkap = np.zeros(dimfkap)

        # Determine midpoint of fkap
        mid = L + 1
        midpt = mid

        # Compute the kernel weights
        for j in range(L + 1):

            # Note that the mid point (fkap[mid - 1]) receives a weight of 1.
            fkap[mid - 1 + j] = math.exp(-(delta * j / hdisc) ** 2 / 2)

            # Because of the kernel's symmetry, weights in equidistance
            # below and above the midpoint are identical.
            fkap[mid - 1 - j] = fkap[mid - 1 + j]

    # If M discretized local bandwidths are given, midpt is an np.ndarray of
    # length bwdisc.
    else:
        dimfkap = 2 * sum(L) + bwdisc
        fkap = np.zeros(dimfkap)
        mid = L[0] + 1

        midpt = np.zeros(bwdisc)

        for i in range(bwdisc - 1):
            midpt[i] = mid

            # Give point in the middle a weight of 1
            fkap[mid - 1] = 1

            for j in range(L[i] + 1):
                fkap[mid + j - 1] = np.exp(-(delta * j / hdisc[i]) ** 2 / 2)
                fkap[mid - j - 1] = fkap[mid + j - 1]

            mid = mid + L[i] + L[i + 1] + 1

        midpt[bwdisc - 1] = mid
        fkap[mid] = 1

        for j in range(L[bwdisc - 1] + 1):
            fkap[mid + j - 1] = np.exp(-(delta * j / hdisc[bwdisc - 1]) ** 2 / 2)
            fkap[mid - j - 1] = fkap[mid + j - 1]

    return fkap, dimfkap, midpt


def combine_bincounts_kernel_weights(
    xcounts, ycounts, bwdisc, M, pp, ppp, dimfkap, fkap, L, midpt, indic, delta
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
    bwdisc: int
        Number of logarithmically equally-spaced bandwidths
        on which local bandwidths are discretized, to speed up computation.
        For the case where 'bandwidth' is a scalar, bwdisc equals 1.
    M: int
        Gridsize, i.e. number of equally-spaced grid points.
    pp: int
        Number of columns of output array tt, i.e the binned approximation to X'W y.
        Degree + 1 (for degree = 2, pp equals 3).
    ppp: int
        Number of columns of output array ss, i.e. the binned approximation to X'W X.
        2 * degree + 1 (for degree = 2, ppp equals 5).
    dimfkap: int
        Length of 1-D array fkap.
    fkap: np.ndarry
        1-D array of length dimfkap containing
        approximated weights for the Gaussian kernel
        (W in the notation above).
    L: int or np.ndarry of length bwdisc
        Parameter defining the number of times the kernel function
        has to be evaluated.
        For the case where 'bandwidth' is a scalar, L is an integer.
        Note that L < N, where N is the total number of observations.
    midpt: int or np.ndarray of length bwdisc.
        For the case where 'bandwidth' is a scalar, midpt is an integer.
        Midpoint of fkap.
    indic: np.ndarry
        1-D array of length M containing ones.
        These are used as index values.
    delta: float
        Bin width.

    Returns
    -------
    ss: np.ndarry
        Dimensions (M, ppp). Binned approximation to X'W X.
    tt: np.ndarry
        Dimensions (M, pp). Binned approximation to X'W y.
    """
    ss = np.zeros((M, ppp))
    tt = np.zeros((M, pp))

    # Distinguish two cases:
    # a) Bandwidth is a scalar
    if bwdisc == 1:
        for g in range(M):
            if xcounts[g] != 0:
                for j in range(max(0, g - L - 1), min(M, g + L)):

                    # Only consider values within the range of fkap.
                    if (g - j + midpt - 1) in range(dimfkap):
                        if indic[j] == 1:

                            fac = 1

                            ss[j, 0] += xcounts[g] * fkap[g - j + midpt - 1]
                            tt[j, 0] += ycounts[g] * fkap[g - j + midpt - 1]

                            for ii in range(1, ppp):
                                fac = fac * delta * (g - j)

                                ss[j, ii] += xcounts[g] * fkap[g - j + midpt - 1] * fac

                                if ii < pp:
                                    tt[j, ii] += (
                                        ycounts[g] * fkap[g - j + midpt - 1] * fac
                                    )

    # b) Bandwidth is a list or np.ndarray of length M
    else:
        for g in range(M):
            if xcounts[g] != 0:

                # Repeat this process bwdisc times.
                for i in range(bwdisc):
                    for j in range(max(0, g - L[i] - 1), min(M, g + L[i])):

                        # Only consider values within the range of fkap.
                        if (g - j + midpt[i] - 1) in range(dimfkap):
                            if indic[j] == 1:

                                fac = 1

                                ss[j, 0] += xcounts[g] * fkap[g - j + midpt[i] - 1]

                                tt[j, 0] += ycounts[g] * fkap[g - j + midpt[i] - 1]

                                for ii in range(1, ppp):
                                    fac = fac * delta * (g - j)

                                    ss[j, ii] += (
                                        xcounts[g] * fkap[g - j + midpt[i] - 1] * fac
                                    )

                                    if ii < pp:
                                        tt[j, ii] += (
                                            ycounts[g]
                                            * fkap[g - j + midpt[i] - 1]
                                            * fac
                                        )

    return ss, tt


def get_curve_estimator(ss, tt, pp, drv, M):
    """
    This functions solves the locally weighted least-squares regression
    problem and returns an estimator for the v-th derivative of beta_,
    the local polynomial estimator, at all points in the grid.
    """
    Smat = np.zeros((pp, pp))
    Tvec = np.zeros(pp)
    curvest = np.zeros(M)

    for g in range(M):
        for i in range(0, pp):
            for j in range(0, pp):
                indss = i + j
                Smat[i, j] = ss[g, indss]

                Tvec[i] = tt[g, i]

        # Calculate beta_ as the solution to the linear matrix equation
        # X'W X * beta_ = X'W y.
        # Note that Smat and Tvec are binned approximations to X'W X and
        # X'W y, respectively, evaluated at the given grid point.
        beta_ = np.linalg.solve(Smat, Tvec)

        # Obtain curve estimator for the desired derivative of beta_.
        curvest[g] = beta_[drv]

    return curvest


@numba.jit
def is_sorted(a):
    """This function checks if the input array is sorted ascendingly."""
    for i in range(a.size - 1):
        if a[i + 1] < a[i]:
            return False
        return True
