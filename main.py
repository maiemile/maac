# Code by @maiemile

import os
import classification_models
import regressor_models
import sampling

# TODO: need a better implementation for this
pipeline = ["classification", "regression"]

# Make sure the folders where the figures are saved exist
# if not, create the corresponding folders
if not os.path.exists("figures\\confusion_matrices"):
    os.makedirs("figures\\confusion_matrices")
if not os.path.exists("figures\\perf_prof"):
    os.makedirs("figures\\perf_prof")

# TODO: eventually this file should contain all parts of the pipeline
# TODO: remove hardcoded variables from files (beginning with classification and regression model files)


if "sampling" in pipeline:
    sampling.do()

if "classification" in pipeline:
    classification_models.do()

if "regression" in pipeline:
    regressor_models.do()