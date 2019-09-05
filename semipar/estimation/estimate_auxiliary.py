"""
This module provides auxiliary functions for the semiparametric 
estimation process.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from skmisc.loess import loess
from matplotlib import pyplot


def estimate_treatment_propensity(D, Z, logit, show_output):
    """
    This function estimates the propensity of selecting into treatment 
    for both treated and untreated individuals based on instruments Z.
    Z subsumes all the observable components that influence the treatment 
    decision, e.g. the decision to enroll into college (D = 1) or not (D = 0).
    
    Estimate propensity scores via Logit (default) or Probit.
    """
    if logit is True:
        logitRslt = sm.Logit(D, Z).fit(disp=0)
        ps = logitRslt.predict(Z)

        if show_output is True:
            print(logitRslt.summary())

    else:
        probitRslt = sm.Probit(D, Z).fit(disp=0)
        ps = probitRslt.predict(Z)

        if show_output is True:
            print(probitRslt.summary())

    return ps.values


def define_common_support(ps, indicator, data, nbins=25, show_output=True):
    """
    This function defines the common support as the region under the histograms
    where propensities in the treated and untreated subsample overlap.
    
    Carneiro et al (2011) choose 25 bins for a total sample of 1747 
    observations, so nbins=25 is set as a default.
    """
    data["ps"] = ps

    treated = data[[indicator, "ps"]][data[indicator] == 1].values
    untreated = data[[indicator, "ps"]][data[indicator] == 0].values

    fig = pyplot.figure()
    hist_treated = pyplot.hist(
        treated[:, 1], bins=nbins, alpha=0.55, label="treated", density=False
    )
    hist_untreated = pyplot.hist(
        untreated[:, 1], bins=nbins, alpha=0.55, label="untreated", density=False
    )

    if show_output is True:
        pyplot.legend(loc="upper center")
        pyplot.grid(axis="y", alpha=0.5)
        pyplot.xlabel("Value")
        pyplot.ylabel("Frequency")
        pyplot.title("Support of P(Z) for D=1 and D=0")
        fig

    else:
        pyplot.close(fig)

    # Find lower limit in the treated subsample.
    # In the treated subsample (D = 1), one expects there to be more people
    # with high propensity scores (ps approaching 1) than indiviudals with
    # low scores (ps approaching 0).
    # Hence, bins closer to zero tend to be smaller and are more likely to
    # be empty than bins on the upper end of the distribution.
    # Now, imagine a case where more than one bin is empty.
    # Let's say one around 0.1 and another at 0.2.
    # Running the for-loop from 1 to 0 guarantees that we find the true lower
    # limit first, i.e. the one at 0.2, which is the "more binding" one.
    # The opposite holds for the untreated sample.

    # Define the lower limit of the common support as the lowest propensity
    # observed in the treated sample, unless one of the histogram bins
    # is empty. In the latter case, take the upper end of that bin as the
    # limit.
    for l in reversed(range(len(hist_treated[0]))):
        if hist_treated[0][l] > 0:
            lower_limit = np.min(treated[:, 1])

        else:
            # print("Premature lower limit found!")
            lower_limit = hist_untreated[1][l + 1]

            break

    # Define the upper limit of the common support as the lowest propensity
    # observed in the untreated sample, unless one of the histogram bins
    # is empty. In the latter case, take the bottom end of that bin as the
    # limit.
    for u in range(len(hist_untreated[0])):
        if hist_untreated[0][u] > 0:
            upper_limit = np.max(untreated[:, 1])

        else:
            # print("Premature upper limit found!")
            upper_limit = hist_untreated[1][u]

            break

    common_support = [lower_limit, upper_limit]

    if show_output is True:
        print(
            """
    Common support lies beteen: 

        {0} and
        {1}""".format(
                lower_limit, upper_limit
            )
        )

    return treated, untreated, common_support


def trim_data(ps, common_support, data, show_output=True):
    """This function trims the data below and above the common support."""
    data_trim = data[(data.ps >= common_support[0]) & (data.ps <= common_support[1])]
    ps_trim = ps[(ps >= common_support[0]) & (ps <= common_support[1])]

    #    if show_output is True:
    #        print("""
    #              {0} observations ({1} percent of the sample)
    #              have been deleted.""". format(
    #                      data.shape[0] - data_trim.shape[0],
    #                      100 * np.round((data.shape[0] - data_trim.shape[0]) / data.shape[0], 4)
    #                      ))

    return data_trim, ps_trim


def construct_Xp(X, ps):
    """
    This function generates the X * ps regressor.
    """
    # To multiply each elememt in X (shape N x k) with the corresponding ps,
    # set up a ps matrix of same size.
    N = len(X)
    ps = pd.Series(ps)
    P_z = pd.concat([ps] * len(X.columns), axis=1, ignore_index=True)

    # Construct Xp
    Xp = pd.DataFrame(
        X.values * P_z.values, columns=[key_ + "_ps" for key_ in list(X)], index=X.index
    )

    return Xp


def generate_residuals(exog, endog, bandwidth=0.05):
    """ 
    This function runs a series of separate loess regressions of a set of
    outcome variables (endog) on a single explanatory variable (exog) and 
    computes the corresponding residuals.
    """
    # Turn input data into np.ndarrays.
    exog = np.array(exog)
    endog = np.array(endog)

    # Determine number of observations and number of columns for the
    # outcome variable.
    N = len(endog)
    col_len = len(endog[0])

    res = np.zeros([N, col_len])

    for i in range(col_len):
        yfit = loess(exog, endog[:, i], span=bandwidth, degree=1)
        yfit.fit()
        res[:, i] = yfit.outputs.fitted_residuals

    return res


def double_residual_reg(ps, X, Xp, Y, rbandwidth, show_output):
    """
    This function performs a Double Residual Regression of X, Xp, and Y on ps.
    
    The LOESS (Locally Estimated Scatterplot Smoothing) method is implemented
    to perform the local linear fit and generate the residuals.
    """
    # 1) Fit a separate local linear regression of X, Xp, and Y on ps,
    # which yields residuals e_X, e_Xp, and e_Y.
    res_X = generate_residuals(ps, X, rbandwidth)
    res_Xp = generate_residuals(ps, Xp, rbandwidth)
    res_Y = generate_residuals(ps, Y, rbandwidth)

    # Append res_X and res_Xp.
    col_names = list(X) + list(Xp)
    res_X_Xp = pd.DataFrame(np.append(res_X, res_Xp, axis=1), columns=col_names)

    # 2) Run a single OLS regression of e_Y on e_X and e_Xp without intercept:
    # e_Y = e_X * beta_0 + e_Xp * (beta_1 - beta_0),
    # to estimate the values of beta_0 and (beta_1 - beta_0).
    model = sm.OLS(res_Y, res_X_Xp)
    results = model.fit()
    b0 = results.params[: len(list(X))]
    b1_b0 = results.params[len((list(X))) :]

    if show_output is True:
        print(results.summary())

    return b0, b1_b0
