#!/usr/bin/env python
"""This script manages all tasks for the TRAVIS build server."""
import os
import subprocess as sp

# if __name__ == "__main__":
#     cmd = [
#         "jupyter",
#         "nbconvert",
#         "--execute",
#         "thesis_notebook.ipynb",
#         "--ExecutePreprocessor.timeout=-1",
#     ]
#     sp.check_call(cmd)

if __name__ == "__main__":
    os.chdir("promotion/thesis_notebook")
    cmd = [
        "jupyter",
        "nbconvert",
        "--execute",
        "thesis_notebook_copy.ipynb",
        "--ExecutePreprocessor.timeout=-1",
    ]
    sp.check_call(cmd)
    os.chdir("../..")