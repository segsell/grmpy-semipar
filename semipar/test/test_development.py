"""
This module contains a simple test for the development process to ensure that
the replicated results of Carneiro et al (2011) remain unaltered after
code optimization.
"""
# import pytest
import pandas as pd

from numpy.testing import assert_equal
from semipar.estimation.estimate import semipar_fit

import sys
sys.path.append("..")


def test_replication():
    """
    This function asserts equality between the test mte_u and the replicated
    mte_u, which is the first derivative of a local polynomial estimator
    of degree 2.
    """
    quantiles, test_mte_u, test_mte_x, test_mte = semipar_fit(
        "files/replication.test.yml"
    )

    expected_mte_u = pd.read_pickle("data/replication-results-mte_u.pkl")
    # expected_mte = pd.read_pickle("data/replication-results-mte.pkl")

    assert_equal(test_mte_u, expected_mte_u)
    # assert_equal(test_mte, expected_mte)


if __name__ == "__main__":
    test_replication()
    print("Everything passed ")
