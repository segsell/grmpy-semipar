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
        Array determining the number of times the kernel function
        has to be evaluated.
        Note that L < N, where N is the total number of observations.
    indic: np.ndarry
        Array of length M containing index values.
    """
    indic = np.ones(M)
    Q = bwdisc

    # tau is chosen so that the interval [-tau, tau] is the
    # "effective support" of the Gaussian kernel,
    # i.e. K is effectively zero outside of [-tau, tau].
    # Opposed to kernel functions with bounded support,
    # the Gaussian kernel, which has infinite support, is merely
    # "effectctively zero" beyond tau,
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
        # to each member of "bandwidth
        if Q > 1:
            log_hdisc = np.log(hdisc)
            gap = (log_hdisc[Q - 1] - log_hdisc[0]) / (Q - 1)
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
        raise Warning(
            "'bandwidth' must be a scalar or np.ndarray " "of length 'gridsize'"
        )

    if min(L) == 0:
        raise Warning(
            "Binning grid too coarse for current small bandwidth: "
            "Consider increasing 'gridsize' M!"
        )

    return hdisc, L, indic


def get_kernel_weights(L, Q, delta, hdisc):
    """This function computes approximated weights for the Gaussian kernel."""
    # If bandwidth is a scalar, midpt is an integer
    if Q == 1:
        dimfkap = 2 * L + Q
        fkap = np.zeros(dimfkap)

        # Determine midpoint of fkap
        mid = L + 1
        midpt = mid

        # Compute the kernel weights
        for j in range(L + 1):

            # Note that the mid point (fkap[mid - 1]) receives a weight of 1.
            fkap[mid - 1 + j] = math.exp(-(delta * j / hdisc) ** 2 / 2)

            # Because of the kernel's symmetry, weights in euqidistance
            # below and above the midpoint are identical.
            fkap[mid - 1 - j] = fkap[mid - 1 + j]

    # If M discretized local bandwidths are given, midpt is an np.ndarray of
    # length Q.
    else:
        dimfkap = 2 * sum(L) + Q
        fkap = np.zeros(dimfkap)
        mid = L[0] + 1

        midpt = np.zeros(Q)

        for i in range(Q - 1):
            midpt[i] = mid

            # Give point in the middle a weight of 1
            fkap[mid - 1] = 1

            for j in range(L[i] + 1):
                fkap[mid + j - 1] = np.exp(-(delta * j / hdisc[i]) ** 2 / 2)
                fkap[mid - j - 1] = fkap[mid + j - 1]

            mid = mid + L[i] + L[i + 1] + 1

        midpt[Q - 1] = mid
        fkap[mid] = 1

        for j in range(L[Q - 1] + 1):
            fkap[mid + j - 1] = np.exp(-(delta * j / hdisc[Q - 1]) ** 2 / 2)
            fkap[mid - j - 1] = fkap[mid + j - 1]

    return fkap, dimfkap, midpt


def combine_bincounts_kernel_weights(
    xcnts, ycnts, Q, M, pp, ppp, dimfkap, fkap, L, midpt, indic, delta
):
    """
    This function combines the bin counts/averages and kernel weights via a
    series of direct convolutions. As a result, binned approximations to
    X'W X and X'W y, denoted by ss and tt, respectively, are computed.
    """
    ss = np.zeros((M, ppp))
    tt = np.zeros((M, pp))

    # Distinguish two cases:
    # a) Bandwidth is a scalar
    if Q == 1:
        for g in range(M):
            if xcnts[g] != 0:
                for j in range(max(0, g - L - 1), min(M, g + L)):

                    # Only consider values within the range of fkap.
                    if (g - j + midpt - 1) in range(dimfkap):
                        if indic[j] == 1:

                            fac = 1

                            ss[j, 0] += xcnts[g] * fkap[g - j + midpt - 1]
                            tt[j, 0] += ycnts[g] * fkap[g - j + midpt - 1]

                            for ii in range(1, ppp):
                                fac = fac * delta * (g - j)

                                ss[j, ii] += xcnts[g] * fkap[g - j + midpt - 1] * fac

                                if ii < pp:
                                    tt[j, ii] += (
                                        ycnts[g] * fkap[g - j + midpt - 1] * fac
                                    )

    # b) Bandwidth is a list or np.ndarray of length M
    else:
        for g in range(M):
            if xcnts[g] != 0:

                # Repeat this process Q times.
                for i in range(Q):
                    for j in range(max(0, g - L[i] - 1), min(M, g + L[i])):

                        # Only consider values within the range of fkap.
                        if (g - j + midpt[i] - 1) in range(dimfkap):
                            if indic[j] == 1:

                                fac = 1

                                ss[j, 0] += xcnts[g] * fkap[g - j + midpt[i] - 1]

                                tt[j, 0] += ycnts[g] * fkap[g - j + midpt[i] - 1]

                                for ii in range(1, ppp):
                                    fac = fac * delta * (g - j)

                                    ss[j, ii] += (
                                        xcnts[g] * fkap[g - j + midpt[i] - 1] * fac
                                    )

                                    if ii < pp:
                                        tt[j, ii] += (
                                            ycnts[g] * fkap[g - j + midpt[i] - 1] * fac
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
    """This function checks if input array is sorted ascendingly."""
    for i in range(a.size - 1):
        if a[i + 1] < a[i]:
            return False
        return True
