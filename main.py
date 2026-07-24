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
import generate_database
import utils as util

from desdeo.emo import (
    algorithms,crossover,mutation
    )


if __name__ == "__main__":

    num_of_repeats = [10]
    num_of_evaluations = [10000]

    # TODO: need a better implementation for this
    pipeline = ["generate_database"]#["approx_pf", "generate_database", "run_experiments", "calculate_indicators"]
    #pipeline = ["classification", "regression"]

    indicators = ["igd", "igd_plus"]

    # Make sure the folders where the figures are saved exist
    # if not, create the corresponding folders
    paths = ["archived_pops", "archived_final_pops", "approx_pfs", os.path.join("figures", "confusion_matrices"), 
             os.path.join("figures", "perf_prof")]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

    options = {"selection": ["TEXT", {
                    "IBEA": algorithms.ibea_options(),
                    "NSGA-III": algorithms.nsga3_options(),
                    "RVEA": algorithms.rvea_options()
                    }], 
           "crossover": ["TEXT", {
                    "SBX":crossover.SimulatedBinaryCrossoverOptions(xover_probability=1.0),
                    "SAX":crossover.SingleArithmeticCrossoverOptions(xover_probability=1.0),
                    "LX":crossover.LocalCrossoverOptions(xover_probability=1.0),
                    "BLX":crossover.BlendAlphaCrossoverOptions(xover_probability=1.0)
                    }], 
           "mutation": ["TEXT", {
                    "BPM":mutation.BoundedPolynomialMutationOptions(),
                    "MPTM":mutation.MPTMutationOptions(),
                    "NUM":mutation.NonUniformMutationOptions(max_generations=100), 
                    "PM":mutation.PowerMutationOptions()
                    }]
        }
    
    # all the problems instances
    problem_instances = [
        # DTLZ problems (3, 4, 6 and 9 objectives, 4 * 7 = 28 instances)
        ["dtlz1", 3, 7],["dtlz2", 3, 10],["dtlz3", 3, 10],["dtlz4", 3, 10],["dtlz5", 3, 10],["dtlz6", 3, 10],["dtlz7", 3, 10],
        ["dtlz1", 4, 11],["dtlz2", 4, 15],["dtlz3", 4, 15],["dtlz4", 4, 15],["dtlz5", 4, 15],["dtlz6", 4, 15],["dtlz7", 4, 15],
        ["dtlz1", 6, 11],["dtlz2", 6, 15],["dtlz3", 6, 15],["dtlz4", 6, 15],["dtlz5", 6, 15],["dtlz6", 6, 15],["dtlz7", 6, 15],
        ["dtlz1", 9, 11],["dtlz2", 9, 15],["dtlz3", 9, 15],["dtlz4", 9, 15],["dtlz5", 9, 15],["dtlz6", 9, 15],["dtlz7", 9, 15],
        # WFG problems (3, 4, 6 and 9 objectives, 4 * 9 = 36 instances)
        ["wfg1", 3, 10],["wfg2", 3, 10],["wfg3", 3, 10],["wfg4", 3, 10],["wfg5", 3, 10],["wfg6", 3, 10],["wfg7", 3, 10],["wfg8", 3, 10],["wfg9", 3, 10],
        ["wfg1", 4, 15],["wfg2", 4, 14], ["wfg3", 4, 14],["wfg4", 4, 15],["wfg5", 4, 15],["wfg6", 4, 15],["wfg7", 4, 15],["wfg8", 4, 15],["wfg9", 4, 15], #WFG2 and WFG3 require an even number of decision variables
        ["wfg1", 6, 15],["wfg2", 6, 14], ["wfg3", 6, 14],["wfg4", 6, 15],["wfg5", 6, 15],["wfg6", 6, 15],["wfg7", 6, 15],["wfg8", 6, 15],["wfg9", 6, 15],
        ["wfg1", 9, 18],["wfg2", 9, 18], ["wfg3", 9, 18],["wfg4", 9, 18],["wfg5", 9, 18],["wfg6", 9, 18],["wfg7", 9, 18],["wfg8", 9, 18],["wfg9", 9, 18],
        # RE problems, 8 instances
        ["re31", 3, 3],["re32", 3, 4],["re33", 3, 4],["re34", 3, 5],["re37", 3, 4],["re41", 4, 7],["re42", 4, 6],["re61", 6, 3], 
    ]

    setup = util.ExperimentalSetup(options, problem_instances)

    # TODO: there may be other paths that need to be checked
    # TODO: remove hardcoded variables from files
    # TODO: add params

    if "generate_database" in pipeline:
        generate_database.do(setup, indicators=indicators, n_of_repeats=num_of_repeats, target_evals=num_of_evaluations)

    if "run_experiments" in pipeline:
       run_experiments.do(setup)

    if "approx_pf" in pipeline:
       pf_approx_from_archives.do()

    if "calculate_indicators" in pipeline:
       calc_indicator_values.do(indicators)

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