import statsmodels.api as sm
import pandas as pd
import numpy as np

from semipar.KernReg.locpoly import locpoly
from scipy.optimize import minimize
from sklearn.utils import resample
from matplotlib import pyplot
from scipy.stats import norm
from scipy.ndimage.filters import gaussian_filter1d

from semipar.estimation.estimate_auxiliary import estimate_treatment_propensity
#from semipar.estimation.estimate_auxiliary import define_common_support
from semipar.estimation.estimate_auxiliary import double_residual_reg
from semipar.estimation.estimate_auxiliary import construct_Xp
from semipar.estimation.estimate_auxiliary import trim_data

from grmpy.estimate.estimate_auxiliary import backward_transformation
from grmpy.estimate.estimate_auxiliary import minimizing_interface
from grmpy.estimate.estimate_auxiliary import calculate_criteria
from grmpy.estimate.estimate_auxiliary import optimizer_options
from grmpy.check.check import check_presence_estimation_dataset
from grmpy.estimate.estimate_output import write_comparison
from grmpy.estimate.estimate_auxiliary import adjust_output
from grmpy.estimate.estimate_auxiliary import start_values
from grmpy.estimate.estimate_auxiliary import process_data
from grmpy.estimate.estimate_output import calculate_mte
from grmpy.estimate.estimate_output import print_logfile
from grmpy.estimate.estimate_auxiliary import bfgs_dict
from grmpy.check.check import check_initialization_dict
from grmpy.check.check import check_presence_init
from grmpy.check.auxiliary import read_data
from grmpy.read.read import read


def bootstrap(init_file, nbootstraps):
    """
    This function generates bootsrapped standard errors
    given an init_file and the number of bootsraps to be drawn.
    """
    check_presence_init(init_file)

    dict_ = read(init_file)
    # np.random.seed(dict_["SIMULATION"]["seed"]) # needed?

    # check_presence_estimation_dataset(dict_)
    # check_initialization_dict(dict_)

    nbins = dict_["ESTIMATION"]["nbins"]
    trim = dict_["ESTIMATION"]["trim_support"]
    rbandwidth = dict_["ESTIMATION"]["rbandwidth"]
    bandwidth = dict_["ESTIMATION"]["bandwidth"]
    gridsize = dict_["ESTIMATION"]["gridsize"]
    a = dict_["ESTIMATION"]["ps_range"][0]
    b = dict_["ESTIMATION"]["ps_range"][1]

    logit = dict_["ESTIMATION"]["logit"]
    show_output = dict_["ESTIMATION"]["show_output"]

    # Distribute initialization information.
    data = read_data(dict_["ESTIMATION"]["file"])

    # Prepare empty arrays to store output values
    mte_boot = np.zeros([gridsize, nbootstraps])
    # quantiles = np.zeros([gridsize, nbootstraps])

    # for i in range(nbootstraps):

    counter = 0
    while counter < nbootstraps:
        boot = resample(data, replace=True, n_samples=len(data), random_state=None)

        # Process data for the semiparametric estimation.
        indicator = dict_["ESTIMATION"]["indicator"]
        D = boot[indicator].values
        Z = boot[dict_["CHOICE"]["order"]]

        # The Local Instrumental Variables (LIV) approach

        # 1. Estimate propensity score P(z)
        ps = estimate_treatment_propensity(D, Z, logit, show_output)


        if isinstance(ps, np.ndarray): #& (np.min(ps) <= 0.3) & (np.max(ps) >= 0.7):

            # 2a. Find common support
            treated, untreated, common_support = define_common_support(
                ps, indicator, boot, nbins, show_output
            )

            # 2b. Trim the data
            if trim is True:
                boot, ps = trim_data(ps, common_support, boot, show_output)

            # 3. Double Residual Regression
            # Sort data by ps
            boot = boot.sort_values(by="ps", ascending=True)
            ps = np.sort(ps)

            X = boot[dict_["TREATED"]["order"]]
            Xp = construct_Xp(X, ps)
            Y = boot[[dict_["ESTIMATION"]["dependent"]]]

            b0, b1_b0 = double_residual_reg(ps, X, Xp, Y, rbandwidth, show_output)

            # Turn the X, Xp, and Y DataFrames into np.ndarrays
            X_arr = np.array(X)
            Xp_arr = np.array(Xp)
            Y_arr = np.array(Y).ravel()

            # 4. Compute the unobserved part of Y
            Y_tilde = Y_arr - np.dot(X_arr, b0) - np.dot(Xp_arr, b1_b0)

            # 5. Estimate mte_u, the unobserved component of the MTE,
            # through a locally quadratic regression
            quantiles, mte_u = locpoly(ps, Y_tilde, 1, 2, bandwidth, gridsize, a, b)

            # 6. construct MTE
            # Calculate the MTE component that depends on X
            mte_x = np.dot(X, b1_b0).mean(axis=0)

            # Put the MTE together
            mte = mte_x + mte_u

            mte_boot[:, counter] = mte

            counter += 1

        else:
            continue

    return mte_boot

def par_fit(init_file):
    """The function estimates the coefficients of the simulated data set."""
    check_presence_init(init_file)

    dict_ = read(init_file)
    np.random.seed(dict_["SIMULATION"]["seed"])

    # We perform some basic consistency checks regarding the user's request.
    check_presence_estimation_dataset(dict_)
    #check_initialization_dict2(dict_)
    #check_init_file(dict_)

    # Distribute initialization information.
    data = read_data(dict_["ESTIMATION"]["file"])
    num_treated = dict_["AUX"]["num_covars_treated"]
    num_untreated = num_treated + dict_["AUX"]["num_covars_untreated"]

    _, X1, X0, Z1, Z0, Y1, Y0 = process_data(data, dict_)

    if dict_["ESTIMATION"]["maxiter"] == 0:
        option = "init"
    else:
        option = dict_["ESTIMATION"]["start"]

    # Read data frame

    # define starting values
    x0 = start_values(dict_, data, option)
    opts, method = optimizer_options(dict_)
    dict_["AUX"]["criteria"] = calculate_criteria(dict_, X1, X0, Z1, Z0, Y1, Y0, x0)
    dict_["AUX"]["starting_values"] = backward_transformation(x0)
    rslt_dict = bfgs_dict()
    if opts["maxiter"] == 0:
        rslt = adjust_output(None, dict_, x0, X1, X0, Z1, Z0, Y1, Y0, rslt_dict)
    else:
        opt_rslt = minimize(
            minimizing_interface,
            x0,
            args=(dict_, X1, X0, Z1, Z0, Y1, Y0, num_treated, num_untreated, rslt_dict),
            method=method,
            options=opts,
        )
        rslt = adjust_output(
            opt_rslt, dict_, opt_rslt["x"], X1, X0, Z1, Z0, Y1, Y0, rslt_dict
        )
    # Print Output files
    print_logfile(dict_, rslt)

    if "comparison" in dict_["ESTIMATION"].keys():
        if dict_["ESTIMATION"]["comparison"] == 0:
            pass
        else:
            write_comparison(data, rslt)
    else:
        write_comparison(data, rslt)

    return rslt


def parametric_mte(rslt, file):
    """This function calculates the marginal treatment effect for different quartiles
    of the unobservable V based on the calculation results."""
    init_dict = read(file)
    data_frame = pd.read_pickle(init_dict["ESTIMATION"]["file"])

    # Define quantiles and read in the original results
    quantiles = [0.0001] + np.arange(0.01, 1.0, 0.01).tolist() + [0.9999]

    # Calculate the MTE and confidence intervals
    mte = calculate_mte(rslt, data_frame, quantiles)
    mte_up, mte_d = calculate_cof_int(rslt, init_dict, data_frame, mte, quantiles)

    return quantiles, mte, mte_up, mte_d



def calculate_cof_int(rslt, init_dict, data_frame, mte, quantiles):
    """This function calculates the confidence intervals of
    the parametric marginal treatment effect.
    """
    # Import parameters and inverse hessian matrix
    hess_inv = rslt["AUX"]["hess_inv"] / data_frame.shape[0]
    params = rslt["AUX"]["x_internal"]
    numx = len(init_dict["TREATED"]["order"]) + len(init_dict["UNTREATED"]["order"])

    # Distribute parameters
    dist_cov = hess_inv[-4:, -4:] * (-1)
    param_cov = hess_inv[:numx, :numx] * (-1)
    dist_gradients = np.array([params[-4], params[-3], params[-2], params[-1]])

    # Process data
    covariates = init_dict["TREATED"]["order"]
    x = np.mean(data_frame[covariates]).tolist()
    x_neg = [-i for i in x]
    x += x_neg
    x = np.array(x)

    # Create auxiliary parameters
    part1 = np.dot(x, np.dot(param_cov, x))
    part2 = np.dot(dist_gradients, np.dot(dist_cov, dist_gradients))
    # Prepare two lists for storing the values
    mte_up = []
    mte_d = []

    # Combine all auxiliary parameters and calculate the confidence intervals
    for counter, i in enumerate(quantiles):
        value = part2 * (norm.ppf(i)) ** 2
        aux = np.sqrt(part1 + value)
        mte_up += [mte[counter] + norm.ppf(0.95) * aux]
        mte_d += [mte[counter] - norm.ppf(0.95) * aux]

    return mte_up, mte_d


def par_weights(rslt, mte, quantiles, file, smooth_tt=0, smooth_tut=0, output=None):
    """This function generates weights for the conventional treatment parameters
    ATE, TT, and TUT."""

    init_dict = read(file)
    data_frame = pd.read_pickle(init_dict["ESTIMATION"]["file"])

    indicator = init_dict["ESTIMATION"]["indicator"]
    Z = data_frame[init_dict["CHOICE"]["order"]]

    logit = init_dict["ESTIMATION"]["logit"]

    if logit is True:
        logitRslt = sm.Logit(data_frame[indicator], Z).fit(disp=0)
        gamma = logitRslt.params
        ps = logitRslt.predict(Z)

    else:
        probitRslt = sm.Probit(data_frame[indicator], Z).fit(disp=0)
        gamma = probitRslt.params
        ps = probitRslt.predict(Z)

    propensity_mean = np.mean(ps)

    dist = rslt['AUX']['x_internal'][-4:]
    cov1v = dist[0] * dist[1]
    cov0v = dist[2] * dist[3]

    tt = []
    tut = []
    ols = []

    aux2 = norm.cdf(np.dot(gamma, Z.T))

    for i in quantiles:
        tt += [(len([j for j in aux2 if j > i]) / Z.shape[0]) / (propensity_mean)]
        tut += [(1 - len([j for j in aux2 if j > i]) / Z.shape[0]) / (1 - propensity_mean)]

    for counter, i in enumerate(quantiles):
        if mte[counter] == 0:
            ols += 0
        else:
            ols += [1 + ((cov1v * norm.ppf(i) * tt[counter] - cov0v * norm.ppf(i) * tut[counter]) / mte[counter])]

    pyplot.rcParams['figure.figsize'] = [17.5, 10]
    ax1 = pyplot.figure().add_subplot(111)
    ax1.set_ylabel(r"$B^{MTE}$", fontsize=24)
    ax1.set_xlabel("$u_D$", fontsize=24)
    ax1.tick_params(axis='both', which='major', labelsize=18)
    ax1.set_ylim([-0.5, 0.55])

    l1 = ax1.plot(quantiles, mte, color='blue', label=' $MTE$', linewidth=3.0)

    tt_arr = np.sort(np.array(tt))[::-1]
    tt_s = gaussian_filter1d(tt_arr, sigma=smooth_tt)

    tut_arr = np.sort(np.array(tut))
    tut_s = gaussian_filter1d(tut_arr, sigma=smooth_tut)

    ate_s = [1.0] * len(tt_s)

    xs = np.arange(0.001, 1., (1 - 0.001) / len(tt_s)).tolist()

    ax2 = ax1.twinx()
    ax2.tick_params(axis='both', which='major', labelsize=18)

    l2 = ax2.plot(xs, tt_s, color='red', linestyle='--', label=' $\omega^{TT}$', linewidth=3.0)
    l3 = ax2.plot(xs, tut_s, color='green', linestyle='--', label=' $\omega^{TUT}$', linewidth=3.0)
    l4 = ax2.plot(xs, ate_s, color='orange', linestyle='-.', label=' $\omega^{ATE}$', linewidth=3.0)

    ax2.set_xlim([-0.005, 1.005])
    ax2.set_ylabel(r"$\omega(u_D)$", fontsize=24)
    pyplot.tight_layout()

    lns = l1 + l2 + l3 + l4
    labs = [l.get_label() for l in lns]

    ax1.legend(lns, labs, loc=0, prop={"size": 24}, frameon=False)
    pyplot.savefig('{}.png'.format(output), dpi=300)

    return tt, tut


# def estimate_treatment_propensity(D, Z, logit, show_output):
#     """
#     This function estimates the propensity of selecting into treatment
#     for both treated and untreated individuals based on instruments Z.
#     Z subsumes all the observable components that influence the treatment
#     decision, e.g. the decision to enroll into college (D = 1) or not (D = 0).
#
#     Estimate propensity scores via Logit (default) or Probit.
#     """
#     if logit is True:
#         logitRslt = sm.Logit(D, Z).fit(disp=0)
#         ps = logitRslt.predict(Z)
#
#         if show_output is True:
#             print(logitRslt.summary())
#
#     else:
#         probitRslt = sm.Probit(D, Z).fit(disp=0)
#         ps = probitRslt.predict(Z)
#
#         if show_output is True:
#             print(probitRslt.summary())
#
#     return ps.values
#
#
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

        # else:
        #     print("Lower limit found!")
        #     lower_limit = hist_untreated[1][l + 1]
        #
        #     break


    # Define the upper limit of the common support as the lowest propensity
    # observed in the untreated sample, unless one of the histogram bins
    # is empty. In the latter case, take the bottom end of that bin as the
    # limit.
    for u in range(len(hist_untreated[0])):
        if hist_untreated[0][u] > 0:
            upper_limit = np.max(untreated[:, 1])

        # else:
        #     print("Upper limit found!")
        #     upper_limit = hist_untreated[1][u]
        #
        #     break

    # lower_limit = np.min(treated[:, 1])
    # upper_limit = np.max(untreated[:, 1])

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


def plot_common_support(init_file, nbins, savefig=True):
    """This function plots histograms of the treated and untreated population
    to assess the common support of the propensity score"""
    dict_ = read(init_file)

    # Distribute initialization information.
    data = read_data(dict_["ESTIMATION"]["file"])

    # Process data for the semiparametric estimation.
    indicator = dict_["ESTIMATION"]["indicator"]
    D = data[indicator].values
    Z = data[dict_["CHOICE"]["order"]]
    logit = dict_["ESTIMATION"]["logit"]

    # estimate propensity score
    ps = estimate_treatment_propensity(D, Z, logit, show_output=False)

    data["ps"] = ps

    treated = data[[indicator, "ps"]][data[indicator] == 1].values
    untreated = data[[indicator, "ps"]][data[indicator] == 0].values

    treated = treated[:, 1].tolist()
    untreated = untreated[:, 1].tolist()

    # Make the histogram using a list of lists
    fig = pyplot.figure(figsize=(13.5, 8))
    hist = pyplot.hist([treated, untreated], bins=nbins,
                    weights=[np.ones(len(treated)) / len(treated), np.ones(len(untreated)) / len(untreated)],
                    density=0,
                    alpha=0.55,
                    label=["Treated", "Unreated"]
                    # stacked=True
                    # color = colors
                    )

    # Plot formatting
    pyplot.legend(loc='upper right')
    pyplot.xticks(np.arange(0, 1.1, step=0.1))
    pyplot.grid(axis='y', alpha=0.25)
    pyplot.xlabel('$P$')
    pyplot.ylabel('$f(P)$')
    pyplot.title('Support of $P(\hat{Z})$ for $D=1$ and $D=0$')

    if savefig is True:
        pyplot.savefig('common_support.png', dpi=300)

    fig.show()


#
#
# def trim_data(ps, common_support, data, show_output=True):
#     """This function trims the data below and above the common support."""
#     data_trim = data[(data.ps >= common_support[0]) & (data.ps <= common_support[1])]
#     ps_trim = ps[(ps >= common_support[0]) & (ps <= common_support[1])]
#
#     #    if show_output is True:
#     #        print("""
#     #              {0} observations ({1} percent of the sample)
#     #              have been deleted.""". format(
#     #                      data.shape[0] - data_trim.shape[0],
#     #                      100 * np.round((data.shape[0] - data_trim.shape[0]) / data.shape[0], 4)
#     #                      ))
#
#     return data_trim, ps_trim
#
#
# def construct_Xp(X, ps):
#     """
#     This function generates the X * ps regressors.
#     """
#     # To multiply each elememt in X (shape N x k) with the corresponding ps,
#     # set up a ps matrix of same size.
#     N = len(X)
#     ps = pd.Series(ps)
#     P_z = pd.concat([ps] * len(X.columns), axis=1, ignore_index=True)
#
#     # Construct Xp
#     Xp = pd.DataFrame(
#         X.values * P_z.values, columns=[key_ + "_ps" for key_ in list(X)], index=X.index
#     )
#
#     return Xp


def generate_residuals_locpol(x, y, bandwidth=0.05):
    """
    This function runs a series of loess regressions for different
    response variables (y) on a single explanatory variable (x)
    and computes the corresponding residuals.
    """
    # Turn input data into np.ndarrays.
    y = np.array(y)
    x = np.array(x)
    
    # Determine number of observations and number of columns for the
    # outcome variable.
    N = len(y)
    col_len = len(y[0])

    res = np.zeros([N, col_len])

    for i in range(col_len):
        quantiles, yfit = locpoly(
            x, y[:, i], derivative=0, degree=1, bandwidth=bandwidth, gridsize=N
        )
        res[:, i] = y[:, i] - yfit

    return res


def double_residual_reg_locpol(ps, X, Xp, Y, rbandwidth, show_output):
    """
    This function performs a Double Residual Regression of X, Xp, and Y on ps.

    A local polynomial function is used to perform the local linear fit
    and generate the residuals.
    """
    # 1) Fit a separate local linear regression of X, Xp, and Y on ps,
    # which yields residuals e_X, e_Xp, and e_Y.
    res_X = generate_residuals_locpol(ps, X, rbandwidth)
    res_Xp = generate_residuals_locpol(ps, Xp, rbandwidth)
    res_Y = generate_residuals_locpol(ps, Y, rbandwidth)

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


def semipar_fit_locpolres(init_file):
    """This functions estimates the MTE via Local Instrumental Variables"""
    check_presence_init(init_file)

    dict_ = read(init_file)
    # np.random.seed(dict_["SIMULATION"]["seed"]) # needed?

    check_presence_estimation_dataset(dict_)
    check_initialization_dict(dict_)

    # Distribute initialization information.
    data = read_data(dict_["ESTIMATION"]["file"])

    # Process data for the semiparametric estimation.
    indicator = dict_["ESTIMATION"]["indicator"]
    D = data[indicator].values
    Z = data[dict_["CHOICE"]["order"]]

    nbins = dict_["ESTIMATION"]["nbins"]
    trim = dict_["ESTIMATION"]["trim_support"]
    rbandwidth = dict_["ESTIMATION"]["rbandwidth"]
    bandwidth = dict_["ESTIMATION"]["bandwidth"]
    gridsize = dict_["ESTIMATION"]["gridsize"]
    a = dict_["ESTIMATION"]["ps_range"][0]
    b = dict_["ESTIMATION"]["ps_range"][1]

    logit = dict_["ESTIMATION"]["logit"]
    show_output = dict_["ESTIMATION"]["show_output"]

    # The Local Instrumental Variables (LIV) approach

    # 1. Estimate propensity score P(z)
    ps = estimate_treatment_propensity(D, Z, logit, show_output)

    # 2a. Find common support
    treated, untreated, common_support = define_common_support(
        ps, indicator, data, nbins, show_output
    )

    # 2b. Trim the data
    if trim is True:
        data, ps = trim_data(ps, common_support, data, show_output)

    # 3. Double Residual Regression
    # Sort data by ps
    data = data.sort_values(by="ps", ascending=True)
    ps = np.sort(ps)

    X = data[dict_["TREATED"]["order"]]
    Xp = construct_Xp(X, ps)
    Y = data[["wage"]]

    b0, b1_b0 = double_residual_reg_locpol(ps, X, Xp, Y, rbandwidth, show_output)

    # Turn the X, Xp, and Y DataFrames into np.ndarrays
    X_arr = np.array(X)
    Xp_arr = np.array(Xp)
    Y_arr = np.array(Y).ravel()

    # 4. Compute the unobserved part of Y
    Y_tilde = Y_arr - np.dot(X_arr, b0) - np.dot(Xp_arr, b1_b0)

    # 5. Estimate mte_u, the unobserved component of the MTE,
    # through a locally quadratic regression
    quantiles, mte_u = locpoly(ps, Y_tilde, 1, 2, bandwidth, gridsize, a, b)

    # 6. construct MTE
    # Calculate the MTE component that depends on X
    mte_x = np.dot(X, b1_b0).mean(axis=0)

    # Put the MTE together
    mte = mte_x + mte_u

    return quantiles, mte_u, mte_x, mte
