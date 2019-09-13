"""This module contains a function that implements the linear binning procedure."""
import numpy as np


def bin_counts_and_averages(x, y, M=401, a=None, delta=None, truncate=True):
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

    If trunctate is True, end observations are truncated.
    Otherwise, weight from end observations is given to corresponding
    end grid points.

    Parameters
    ----------
    x: np.ndarray
        Array of predictor variables.
    y: np.ndarray
        Array of the response variable. One-dimensional.
    M: int
        Length of the grid mesh over which x and y are to be evaluated.
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
    # Turn r-dimensional array into 1-d array.
    if x.ndim > 1:
        x = x.ravel()

    # Set bin width if not given.
    if delta is None:
        a = min(x)
        b = max(x)
        delta = (b - a) / (M - 1)

    n = len(x)

    xcounts = np.zeros(M)
    ycounts = np.zeros(M)
    lxi = np.zeros(n)
    li = np.zeros(n)
    rem = np.zeros(n)

    for i in range(n):
        lxi[i] = ((x[i] - a) / delta) + 1

        # Find integer part of "lxi"
        li[i] = int(lxi[i])
        rem[i] = lxi[i] - li[i]

    for g in range(M):
        # np.where(li == g) returns a tuple with the first element
        # being an np.ndarry containing indeces. These indices denote where
        # an entry in li is equal to the respective gridpoint g.
        if len(np.where(li == g)[0]) > 0:
            # In case more than one entry in li euqals g,
            # go through all of them one by one.
            for j in range(len(np.where(li == g)[0])):

                xcounts[g - 1] += 1 - rem[np.where(li == g)[0][j]]
                xcounts[g] += rem[np.where(li == g)[0][j]]

                # If the predictor variable x is multidimensional and hence
                # len(x) is larger than len(y), consider only values
                # in range of y.
                if (np.where(li == g)[0][j]) in range(len(y)):
                    ycounts[g - 1] += (1 - rem[np.where(li == g)[0][j]]) * y[
                        np.where(li == g)[0][j]
                    ]

                    ycounts[g] += (
                        rem[np.where(li == g)[0][j]] * y[np.where(li == g)[0][j]]
                    )

    # By default, end observations are truncated.
    for i in range(n):
        if truncate is True:
            pass

        # Truncation is implicit if there are no points in li
        # beyond the grid's boundary points.
        elif 1 <= li[i] < M:
            pass

        # If truncate=False, weight from end observations is given to
        # corresponding end grid points.
        elif li[i] < 1 and truncate is False:
            if len(np.where(li < 1)[0]) > 0:
                for j in range(len(np.where(li < 1)[0])):
                    xcounts[0] += 1
                    ycounts[0] += [np.where(li < 1)[0][j]]

        elif li[i] >= M and truncate is False:
            if len(np.where(li == M)[0]) > 0:
                for j in range(len(np.where(li == M)[0])):
                    xcounts[M - 1] += 1
                    ycounts[M - 1] += y[np.where(li == M)[0][j]]

    return xcounts, ycounts
