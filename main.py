# Code by @maiemile

import os
import classification_models
import regressor_models

# Make sure the folders where the figures are saved exist
# if not, create the corresponding folders
if not os.path.exists("figures\\confusion_matrices"):
    os.makedirs("figures\\confusion_matrices")
if not os.path.exists("figures\\perf_prof"):
    os.makedirs("figures\\perf_prof")

# TODO: eventually this file should contain all parts of the pipeline
# TODO: remove hardcoded variables from files (beginning with classification and regression model files)

classification_models.do()
regressor_models.do()