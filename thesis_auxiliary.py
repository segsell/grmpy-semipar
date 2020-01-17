"""This module provides auxiliary functions for the thesis notebook"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from semipar.KernReg.locpoly import locpoly
from scipy.optimize import minimize
from sklearn.utils import resample
from scipy.stats import norm

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
from grmpy.check.check import check_presence_init
from grmpy.check.auxiliary import read_data
from grmpy.read.read import read


def plot_common_support(init_file, nbins, fs=24, output=False):
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
    fig = plt.figure(figsize=(17.5, 10))
    hist = plt.hist(
        [treated, untreated],
        bins=nbins,
        weights=[
            np.ones(len(treated)) / len(treated),
            np.ones(len(untreated)) / len(untreated),
        ],
        density=0,
        alpha=0.55,
        label=["Treated", "Untreated"],
    )

    # Plot formatting
    plt.tick_params(axis="both", labelsize=14)
    plt.legend(loc="upper right", prop={"size": 14})
    plt.xticks(np.arange(0, 1.1, step=0.1))
    plt.grid(axis="y", alpha=0.25)
    plt.xlabel("$P$", fontsize=fs)
    plt.ylabel("$f(P)$", fontsize=fs)
    # plt.title('Support of $P(\hat{Z})$ for $D=1$ and $D=0$', fontsize=fs)

    if not output is False:
        plt.savefig(output, dpi=300)

    fig.show()


def plot_mte_carneiro(rslt, init_file, nbootstraps=250):
    """This function plots the original and the replicated MTE from Carneiro et al. (2011)"""
    # mte per year of university education
    mte = rslt["mte"] / 4
    quantiles = rslt["quantiles"]

    # bootstrap 90 percent confidence bands
    mte_boot = bootstrap(init_file, nbootstraps)

    # mte per year of university education
    mte_boot = mte_boot / 4

    # Get standard error of MTE at each gridpoint u_D
    mte_boot_std = np.std(mte_boot, axis=1)

    # Compute 90 percent confidence intervals
    con_u = mte + norm.ppf(0.95) * mte_boot_std
    con_d = mte - norm.ppf(0.95) * mte_boot_std

    # Load original data
    mte_ = pd.read_csv(
        "semipar/data/mte_semipar_original.csv"
    )

    # Plot
    ax = plt.figure(figsize=(17.5, 10)).add_subplot(111)

    ax.set_ylabel(r"$MTE$", fontsize=20)
    ax.set_xlabel("$u_D$", fontsize=20)
    ax.tick_params(
        axis="both", direction="in", length=5, width=1, grid_alpha=0.25, labelsize=14
    )
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    ax.xaxis.set_ticks(np.arange(0, 1.1, step=0.1))
    ax.yaxis.set_ticks(np.arange(-1.8, 0.9, step=0.2))

    ax.margins(x=0.003)
    ax.margins(y=0.03)

    # Plot replicated curves
    ax.plot(quantiles, mte, label="replicated $MTE$", color="orange", linewidth=4)
    ax.plot(quantiles, con_u, color="orange", linestyle=":", linewidth=3)
    ax.plot(quantiles, con_d, color="orange", linestyle=":", linewidth=3)

    # Plot original curve
    ax.plot(
        mte_["quantiles"],
        mte_["mte"],
        label="$original MTE$",
        color="blue",
        linewidth=4,
    )
    ax.plot(mte_["quantiles"], mte_["con_u"], color="blue", linestyle=":", linewidth=3)
    ax.plot(mte_["quantiles"], mte_["con_d"], color="blue", linestyle=":", linewidth=3)

    ax.set_ylim([-0.77, 0.86])
    ax.set_xlim([0, 1])

    blue_patch = mpatches.Patch(color="orange", label="replicated $MTE$")
    orange_patch = mpatches.Patch(color="blue", label="original $MTE$")
    plt.legend(handles=[blue_patch, orange_patch], prop={"size": 16})

    plt.show()

    return mte, quantiles


def plot_mte_soep(rslt, init_file, nbootstraps=250):
    """This function plots the semiparametric MTE for the soep sample"""
    # mte per year of university education
    mte = rslt["mte"] / 5
    quantiles = rslt["quantiles"]

    # bootstrap 90 percent confidence bands
    np.random.seed(6295)
    mte_boot = bootstrap(init_file, nbootstraps)

    # mte per year of university education
    mte_boot = mte_boot / 5

    # Get standard error of MTE at each gridpoint u_D
    mte_boot_std = np.std(mte_boot, axis=1)

    # Compute 90 percent confidence intervals
    con_u = mte + norm.ppf(0.95) * mte_boot_std
    con_d = mte - norm.ppf(0.95) * mte_boot_std

    # Plot
    ax = plt.figure(figsize=(17.5, 10)).add_subplot(111)

    ax.set_ylabel(r"$MTE$", fontsize=20)
    ax.set_xlabel("$u_D$", fontsize=20)
    ax.tick_params(
        axis="both", direction="in", length=5, width=1, grid_alpha=0.25, labelsize=14
    )
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    ax.xaxis.set_ticks(np.arange(0, 1.1, step=0.1))
    ax.yaxis.set_ticks(np.arange(-1.2, 1.2, step=0.2))

    ax.margins(x=0.005)
    ax.margins(y=0.05)

    #ax.set_ylim([-1.2, 1.2])
    ax.set_xlim([0, 1])

    ax.plot(quantiles, mte, label="$no migrants$", color="orange", linewidth=4)
    ax.plot(quantiles, con_u, color="orange", linestyle=":")
    ax.plot(quantiles, con_d, color="orange", linestyle=":")

    plt.show()

    return mte, quantiles


def bootstrap(init_file, nbootstraps, show_output=False):
    """
    This function generates bootsrapped standard errors
    given an init_file and the number of bootsraps to be drawn.
    """
    check_presence_init(init_file)
    dict_ = read(init_file)

    nbins = dict_["ESTIMATION"]["nbins"]
    trim = dict_["ESTIMATION"]["trim_support"]
    rbandwidth = dict_["ESTIMATION"]["rbandwidth"]
    bandwidth = dict_["ESTIMATION"]["bandwidth"]
    gridsize = dict_["ESTIMATION"]["gridsize"]
    a = dict_["ESTIMATION"]["ps_range"][0]
    b = dict_["ESTIMATION"]["ps_range"][1]

    logit = dict_["ESTIMATION"]["logit"]

    # Distribute initialization information.
    data = read_data(dict_["ESTIMATION"]["file"])

    # Prepare empty arrays to store output values
    mte_boot = np.zeros([gridsize, nbootstraps])

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

        if isinstance(ps, np.ndarray):  # & (np.min(ps) <= 0.3) & (np.max(ps) >= 0.7):

            # 2a. Find common support
            treated, untreated, common_support = define_common_support(
                ps, indicator, boot, nbins, show_output
            )

            # 2b. Trim the data
            if trim is True:
                boot, ps = trim_data(ps, common_support, boot)

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
    # check_initialization_dict2(dict_)
    # check_init_file(dict_)

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


def plot_mte_par(rslt, init_file):
    """This function plots the parametric MTE for the soep sample"""
    quantiles, mte, mte_up, mte_d = parametric_mte(rslt, init_file)

    # Plot
    ax = plt.figure(figsize=(17.5, 10)).add_subplot(111)

    ax.set_ylabel(r"$MTE$", fontsize=20)
    ax.set_xlabel("$u_D$", fontsize=20)
    ax.tick_params(
        axis="both", direction="in", length=5, width=1, grid_alpha=0.25, labelsize=14
    )
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    ax.xaxis.set_ticks(np.arange(0, 1.1, step=0.1))
    ax.yaxis.set_ticks(np.arange(-1.8, 0.9, step=0.2))

    ax.margins(x=0.005)
    ax.margins(y=0.05)

    ax.set_ylim([-0.5, 0.5])
    ax.set_xlim([0, 1])

    ax.plot(
        quantiles,
        [i / 5 for i in mte],
        label="$no migrants$",
        color="purple",
        linewidth=4,
    )
    ax.plot(quantiles, [i / 5 for i in mte_up], color="purple", linestyle=":")
    ax.plot(quantiles, [i / 5 for i in mte_d], color="purple", linestyle=":")

    plt.show()

    return mte, quantiles


def calculate_cof_int(rslt, init_dict, data_frame, mte, quantiles):
    """This function calculates the confidence intervals of
    the parametric marginal treatment effect.
    """
    # Import parameters and inverse hessian matrix
    hess_inv = rslt["AUX"]["hess_inv"] / data_frame.shape[0]
    params = rslt["AUX"]["x_internal"]
    numx = len(init_dict["TREATED"]["order"]) + len(init_dict["UNTREATED"]["order"])

    # Distribute parameters
    dist_cov = hess_inv[-4:, -4:]
    param_cov = hess_inv[:numx, :numx]
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

    fig = plt.figure()
    hist_treated = plt.hist(
        treated[:, 1], bins=nbins, alpha=0.55, label="treated", density=False
    )
    hist_untreated = plt.hist(
        untreated[:, 1], bins=nbins, alpha=0.55, label="untreated", density=False
    )

    if show_output is True:
        plt.legend(loc="upper center")
        plt.grid(axis="y", alpha=0.5)
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.title("Support of P(Z) for D=1 and D=0")
        fig

    else:
        plt.close(fig)

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