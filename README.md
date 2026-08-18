# Modular Automatic Algorithm Configuration (MAAC)

## Get started

For now, you can run the full pipeline. The new full dataset will be available later, which will enable users to use pre-existing configurator models for their problems.

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

#### Database generation

This package uses an SQLite database for storing most relational data. The database structure is created by the generate_database.py file.
The script also fills in the supplied data for EA configurations, problems and run details.

#### Run the experiments

The experiments are run using the file run_experiments.py. Each configuration is run on each problem for each seed and the defined number of function evaluations. The ELA features are calculated once per unique seed on each problem. The feature calculations are performed by using the initial population sampled using the Latin hypercube sampling technique. The non-dominated archive and final population are saved from each run. Only runs without an existing archive are run.

#### Reference Pareto front calculations

The reference Pareto fronts (PFs) are calculated from the non-dominated archives in the file pf_approx_from_archives.py. First, all archives are merged together to fetch the full set of non-dominated solutions across all archives. Then, distance-based subset selection is performed to limit the size of the reference PF.

NOTE: If the merged non-dominated archives would be excessively large, it is recommended to use alternative archiving strategies.

#### Indicator calculations

The performance indicators are calculated using the reference PFs and non-dominated archives in the file calc_indicator_values.py. The values are only calculated for the given indicators if no value exists and only for runs that have both a non-dominated archive and a corresponding reference PF available.  

## Basic guides

### Choose a different indicator

By default, MAAC uses IGD as the performance indicator. If you wish to change the indicator, change the "indicator" field in config.txt.
Furthermore, if you wish to calculate multiple indicators, TODO:
Currently supported indicators (use the name in the brackets): IGD (igd), IGD+ (igd_plus).
Guide for adding other indicators is TODO:

### Add new problems/instances

Currently, the following problem suites are supported natively:
- All DTLZ problems from pymoo
- All WFG problems from pymoo
- 8 RE problems from DESDEO/local implementations (see utils.py)

Adding new problem instances from DTLZ/WFG is simple. Just modify the "problem_instances" object in main.py.
Additionally, if you are running the run_experiments.py file directly, modify the "problem_instances" object.
If you wish to add new problems from pymoo or DESDEO, the process is almost as straightforward.
For pymoo, follow the example set in "get_problem_object" in utils.py.
For DESDEO, follow the examples set in "get_problem_object" and "get_re_problems" in utils.py.
Support for other problems is not planned.

### TODO: Add new parameters/options

### TODO: Change the experimental structure

### TODO: Change the hyperparameter optimization process

### TODO: CSC Roihu support

# Old guide

If you wish to use the old version of the package, download the earliest available version and have a look at the README.md file.