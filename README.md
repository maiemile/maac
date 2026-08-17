# Modular Automatic Algorithm Configuration (MAAC)

## Get started

For now, you can run the full pipeline. The new full dataset will be available later, which will enable users to use pre-existing configurator models
for their problems.

### Required libraries

If you wish to run the entire pipeline of code, some libraries must be installed in your environment:
- DESDEO
- pflacco
- xgboost
- scienceplots
- numdifftools

### The pipeline

The pipeline consists of several components. The individual components are described in further detail below the list.

- main.py: This file exists to streamline the process of running several scripts. It also creates the necessary folders for other scripts.
- generate_database.py: Creates the following tables of the database: EAs, problems, features and runs. Also contains most database query functions.
- run_experiments.py: Runs the configurations on the problems. Saves the final populations and non-dominated archives. Also runs the necessary ELA calculations.
- (sampling.py): Conducts the ELA calculations by sampling the problems. Is automatically run via run_experiments.py.
- pf_approx_from_archives.py: Calculates the reference Pareto fronts per problem from the associated non-dominated archives. Performs distance-based subset selection.
TODO: the code for handling too large archives is missing
- calc_indicator_values.py: Calculates the indicator values of the runs using the reference PFs and non-dominated archives.
- regressor_models.py: Preprocesses data before training and testing regressor-based configurator models. Creates some visualizations on the test results.
- classification_models.py: Preprocesses data before training and testing classifier-based configurator models. Creates some visualizations on the test results.

Other files:
- utils.py: Contains many useful functions used by the other scripts.
- reprob.py: Implements some RE problems missing from DESDEO.
- perfprof.py: Slightly modified version of the performance profile plot code from the [perfprof](https://github.com/dmsteck/perfprof.py) package.
- config.txt: Contains many important configuration options for running the scripts.

### TODO: Roihu support

# Old guide

## Get Started

The other datasets can be found on [Zenodo](https://doi.org/10.5281/zenodo.20393563).

### Required libraries

If you wish to run the entire pipeline of code, some libraries must be installed in your environment:
- DESDEO
- reproblem
- pflacco
- xgboost

### Conduct algorithm configuration runs

The algorithm configurations are run on a set of problems using the file run_experiments.py.

### Calculate exploratory landscape analysis (ELA) features

All ELA features are calculated using the file sampling.py.

### Calculate Pareto front approximations

Pareto front approximations are calculated using the file pf_approx_from_archives.py.
By default, a maximum of 2000 solutions are included in the approximation using distance-based subset selection.

### Calculate performance indicator values

Performance indicator values are calculated with the file calc_indicator_values.py.
By default, IGD and IGD+ are calculated. The file contains code for calculating HV values,
but note that the calculation of HV values is **extremely slow** once the number of objective
functions increases.
Best configurations in terms of IGD, IGD+, and HV can be quickly calculated per problem using best_hv_values.py and best_igd.py

### Run and test configurator models

Classification-based configurator models are run using the file classification_models.py (alternatively, configurator_model.ipynb).
Regression-based configurator models are run using the file regressor_models.py.
These files include code for creating visualizations of the confusion matrices, performance profile plots and decision trees.
R^2 scores of the regressor models can also be calculated in the file regressor_models.py.

If you wish to retrain the configurator models instead of testing existing models, set the variable "load_models" in config.txt to False

### (Optional: Use Puhti for expensive calculations)

UPDATE: Puhti will be decommissioned by the end of July 2026. New scripts for [Roihu](https://docs.csc.fi/computing/systems-roihu/) might be added here later.

Two shell script files for the Puhti supercomputer at CSC are included in this repository:
- experimental_script.sh: runs the main script for running EA configurations on problems
- pf_approx_script.sh: calculates Pareto front approximations from archives

In the script files, fill in spaces marked with 'XXXXXX' with your project name. Python has to be set up with venv.
More information on Puhti can be found here: [https://docs.csc.fi/computing/systems-puhti/](https://docs.csc.fi/computing/systems-puhti/)
