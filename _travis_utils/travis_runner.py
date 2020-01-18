#!/usr/bin/env python
"""This script manages all tasks for the TRAVIS build server."""
import subprocess as sp

# if __name__ == "__main__":
#
#     notebook = "thesis_notebook.ipynb"
#     cmd = " jupyter nbconvert --execute {}  \
#     --ExecutePreprocessor.timeout=-1".format(
#         notebook
#     )
#     sp.check_call(cmd, shell=True)


if __name__ == "__main__":
    cmd = [
        "jupyter",
        "nbconvert",
        "--execute",
        "thesis_notebook.ipynb",
        "--ExecutePreprocessor.timeout=-1",
    ]
    sp.check_call(cmd)