"""
This module provides the semiparametric estimation process.
"""
import numpy as np

from grmpy.check.check import check_presence_estimation_dataset
from grmpy.check.check import check_initialization_dict
from grmpy.check.check import check_presence_init
from grmpy.check.auxiliary import read_data
from grmpy.read.read import read

from semipar.estimation.estimate_auxiliary import estimate_treatment_propensity
from semipar.estimation.estimate_auxiliary import define_common_support
from semipar.estimation.estimate_auxiliary import double_residual_reg
from semipar.estimation.estimate_auxiliary import construct_Xp
from semipar.estimation.estimate_auxiliary import trim_data

from semipar.KernReg.locpoly import locpoly


def semipar_fit(init_file):
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
    reestimate = dict_["ESTIMATION"]["reestimate_p"]
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

    # 2c. Re-estimate baseline propensity score on the trimmed sample
    if reestimate is True:
        D = data[indicator].values
        Z = data[dict_["CHOICE"]["order"]]

        # Re-estimate propensity score P(z)
        ps = estimate_treatment_propensity(D, Z, logit, show_output)

    # 3. Double Residual Regression
    # Sort data by ps
    data = data.sort_values(by="ps", ascending=True)
    ps = np.sort(ps)

    X = data[dict_["TREATED"]["order"]]
    Xp = construct_Xp(X, ps)
    Y = data[[dict_["ESTIMATION"]["dependent"]]]

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

    #X, b1_b0

    return quantiles, mte_u, mte_x, mte
