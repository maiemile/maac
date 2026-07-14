# Code by @maiemile

# This is the main file

import os
import classification_models
import regressor_models
import sampling
import igd_analysis
import best_igd
import run_experiments 
import pf_approx_from_archives
import calc_indicator_values


if __name__ == "__main__":

    num_of_repeats = 10
    num_of_evaluations = 1000

    # TODO: need a better implementation for this
    pipeline = ["run_experiments", "approx_pf", "calculate_indicators"]
    #pipeline = ["classification", "regression"]

    # Make sure the folders where the figures are saved exist
    # if not, create the corresponding folders
    paths = ["archived_pops", "archived_final_pops", "approx_pfs", "figures\\confusion_matrices", "figures\\perf_prof"]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

    # TODO: there may be other paths that need to be checked

    # TODO: remove hardcoded variables from files (beginning with classification and regression model files)

    # TODO: add database generation + params

    if "run_experiments" in pipeline:
       run_experiments.do()

    if "approx_pf" in pipeline:
       pf_approx_from_archives.do()

    if "calculate_indicators" in pipeline:
       calc_indicator_values.do()

    if "best_indicator" in pipeline:
        best_igd.do()

    if "igd_analysis" in pipeline:
        igd_analysis.do()

    if "sampling" in pipeline:
        sampling.do()

    if "classification" in pipeline:
        classification_models.do()

    if "regression" in pipeline:
        regressor_models.do()