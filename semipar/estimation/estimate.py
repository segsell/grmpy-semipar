"""
This module provides the semiparametric estimation process.
"""
import numpy as np

from grmpy.check.check import check_presence_estimation_dataset
from grmpy.check.check import check_presence_init
from grmpy.check.auxiliary import read_data
from grmpy.read.read import read

from grmpy.check.auxiliary import check_special_conf
from grmpy.check.custom_exceptions import UserError
from grmpy.check.auxiliary import is_pos_def


from semipar.estimation.estimate_auxiliary import estimate_treatment_propensity
from semipar.estimation.estimate_auxiliary import define_common_support
from semipar.estimation.estimate_auxiliary import double_residual_reg
from semipar.estimation.estimate_auxiliary import construct_Xp
from semipar.estimation.estimate_auxiliary import trim_data

from semipar.KernReg.locpoly import locpoly

from scipy.optimize import minimize

from grmpy.estimate.estimate_auxiliary import backward_transformation
from grmpy.estimate.estimate_auxiliary import minimizing_interface
from grmpy.estimate.estimate_auxiliary import calculate_criteria
from grmpy.estimate.estimate_auxiliary import optimizer_options
from grmpy.estimate.estimate_output import write_comparison
from grmpy.estimate.estimate_auxiliary import adjust_output
from grmpy.estimate.estimate_auxiliary import start_values
from grmpy.estimate.estimate_auxiliary import process_data
from grmpy.estimate.estimate_output import print_logfile
from grmpy.estimate.estimate_auxiliary import bfgs_dict


def fit(init_file, semipar=False):
    """ """
    check_presence_init(init_file)

    dict_ = read(init_file)

    # Perform some consistency checks given the user's request
    check_presence_estimation_dataset(dict_)
    check_initialization_dict(dict_)

    # Semiparametric Model
    if semipar is True:
        quantiles, mte_u, X, b1_b0 = semipar_fit(init_file)  # change to dict_

        # Construct MTE
        # Calculate the MTE component that depends on X
        mte_x = np.dot(X, b1_b0)

        # Put the MTE together
        mte = mte_x.mean(axis=0) + mte_u

        # Accounting for variation in X
        mte_min = np.min(mte_x) + mte_u
        mte_max = np.max(mte_x) + mte_u

        rslt = {
            "quantiles": quantiles,
            "mte": mte,
            "mte_x": mte_x,
            "mte_u": mte_u,
            "mte_min": mte_min,
            "mte_max": mte_max,
            "X": X,
            "b1-b0": b1_b0,
        }

    # Parametric Normal Model
    else:
        check_par(dict_)
        rslt = par_fit(dict_)

    return rslt


def par_fit(dict_):
    """The function estimates the coefficients of the simulated data set."""
    np.random.seed(dict_["SIMULATION"]["seed"])

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


def check_initialization_dict(dict_):
    """This function performs some basic checks regarding the integrity of the user's
    request. There should be no uncontrolled terminations of the package once these
    checks are passed.
    """
    # Some basic checks
    for key_ in ["TREATED", "UNTREATED", "CHOICE"]:
        if len(dict_[key_]["order"]) > len(set(dict_[key_]["order"])):
            msg = (
                "There is a problem in the {} section of the initialization file. \n"
                "         "
                "Probably you specified two coefficients for one covariate in the "
                "same section.".format(key_)
            )
            raise UserError(msg)

    if dict_["ESTIMATION"]["file"][-4:] not in [".pkl", ".txt", "dta"]:
        msg = (
            "The {} format specified in the Estimation section of the initialization "
            "file is currently not supported by grmpy. \n"
            "         Please use either .txt, .pkl or .dta files.".format(
                dict_["ESTIMATION"]["file"][-4:]
            )
        )
        raise UserError(msg)


def check_par(dict_):
    """"""
    # Distribute details
    num_agents_sim = dict_["SIMULATION"]["agents"]

    # These are examples for a whole host of tests.
    if num_agents_sim <= 0:
        msg = "The number of simulated individuals needs to be larger than zero."
        raise UserError(msg)

    if dict_["DETERMINISTIC"] is False:
        if not is_pos_def(dict_):
            msg = "The specified covariance matrix has to be positive semidefinite."
            raise UserError(msg)

    if all(dist_elements == 0 for dist_elements in dict_["DIST"]["params"]):
        msg = "The distributional characteristics have to be undeterministic."
        raise UserError(msg)
    elif dict_["DIST"]["params"][5] == 0:
        msg = (
            "The standard deviation of the collected unobservables have to be larger"
            " than zero."
        )
        raise UserError(msg)

    for key_ in ["TREATED", "UNTREATED", "CHOICE"]:
        if len(set(dict_[key_]["order"])) != len(dict_[key_]["order"]):
            msg = "There are two start coefficients {} Section".format(key_)
            raise UserError(msg)
        if (
            "params" not in dict_[key_].keys()
            and dict_["ESTIMATION"]["start"] == "init"
        ):
            msg = (
                "The missing of a pre-specified paramterization in the {} section does"
                " not correspond with the start value option of your initialization "
                "file. \n        We recommend to switch to the generation of automatic"
                " start values by changing the start flag in the ESTIMATION section "
                'from "init" to "auto".'.format(key_)
            )
            raise UserError(msg)

    error, msg = check_special_conf(dict_)
    if error is True:
        raise UserError(msg)


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
        data, ps = trim_data(ps, common_support, data)

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
    # mte_x = np.dot(X, b1_b0).mean(axis=0)

    # Put the MTE together
    # mte = mte_x + mte_u

    return quantiles, mte_u, X, b1_b0
