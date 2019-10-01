"""This module contains a function that implements the linear binning procedure."""
import numpy as np
from numba import jit


@jit(nopython=True)
def linear_binning(x, y, M, a, delta, truncate=True):
    """
    This function generates bin counts and bin averages over an equally spaced
    grid via the linear binning strategy.
    In essence, bin counts are obtained by assigning the raw data to
    neighboring grid points. A bin count can be thought of as representing the
    amount of data in the neighborhood of its corresponding grid point.
    Counts on the y-axis display the respective bin averages.

    The linear binning strategy is based on the linear transformation
    lxi = ((x - a) / delta) + 1, calculated for each observation
    x_i, i = 1, ... n. Its integer part, denoted by li, indicates the two
    nearest bin centers to x_i. This calculation already does the trick
    for simple binning. For linear binning, however, we additonally compute the
    "fractional part" or "remainder", lxi - li, which gives the weights
    attached to the two nearest bin centers, namely (1 - rem) for the bin
    considered and rem for the next bin.

    If truncate is True, end observations are truncated.
    Otherwise, weight from end observations is given to corresponding
    end grid points.

    Parameters
    ----------
    x: np.ndarray
        Array of the predictor variable. Shape (N,).
        Missing values are not accepted. Must be sorted.
    y: np.ndarray
        Array of the response variable. Shape (N,).
        Missing values are not accepted. Must come presorted by x.
    M: int
        Gridsize, i.e. number of equally-spaced grid points
        over which x and y are to be evaluated.
    a: float
        Start point of the grid.
    delta: float
        Bin width.
    truncate: bool
        If True, then endpoints are truncated.

    Returns
    -------
    xcounts: np.ndarry
        Array of binned x-values ("bin counts") of length M.
    ycounts: np.ndarry
        Array of binned y-values ("bin averages") of length M.
    """
    n = len(x)

    xcounts = np.zeros(M)
    ycounts = np.zeros(M)
    lxi = np.zeros(n)
    rem = np.zeros(n)
    li = [0] * n

    for i in range(n):
        lxi[i] = ((x[i] - a) / delta) + 1

        # Find integer part of "lxi"
        li[i] = int(lxi[i])
        rem[i] = lxi[i] - li[i]

    for gridpoint in range(M):
        indices = [i for i, element in enumerate(li) if element == gridpoint]

        for index in indices:
            xcounts[gridpoint - 1] += 1 - rem[index]
            xcounts[gridpoint] += rem[index]

            ycounts[gridpoint - 1] += (1 - rem[index]) * y[index]
            ycounts[gridpoint] += rem[index] * y[index]

    # By default, end observations are truncated.
    if truncate is True:
        pass

    # Truncation is implicit if there are no points in li
    # beyond the grid's boundary points.
    # Note that li is sorted. So it is sufficient to check if
    # the conditions below hold for the bottom and top
    # observation, respectively
    elif 1 <= li[0] and li[n - 1] < M:
        pass

    # If truncate=False, weight from end observations is given to
    # corresponding end grid points.
    # elif li[i] < 1 and truncate is False:
    elif truncate is False:
        indices_bottom = [i for i, element in enumerate(li) if element < 1]

        for index in indices_bottom:
            xcounts[0] += 1
            ycounts[0] += y[index]

        # elif li[i] >= M:
        indices_top = [i for i, element in enumerate(li) if element >= M]

        for index in indices_top:
            xcounts[M - 1] += 1
            ycounts[M - 1] += y[index]

    return xcounts, ycounts
