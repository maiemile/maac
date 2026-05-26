# Modular Automatic Algorithm Configuration (MAAC)

## Get Started

### Required libraries

If you wish to run the entire pipeline of code, some libraries must be installed in your environment:
- DESDEO
- reproblem
- pflacco
- xgboost

### Conduct algorithm configuration runs

The algorithm configurations are run on a set of problems using the file puhti_python_script.py.

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

Classification-based configurator models are run using the file configurator_model.ipynb.
Regression-based configurator models are run using the file regressor_models.py.
These files include code for creating visualizations of the confusion matrices, performance profile plots and decision trees.
R^2 scores of the regressor models can also be calculated in the file regressor_models.py.

If you already have configurator models and wish to only test them without training new models, set the following variable values:
- In configurator_model.ipynb: load_models = True
- In regressor_models.py: load_from_files = True

### (Optional: Use Puhti for expensive calculations)

Two shell script files for the Puhti supercomputer at CSC are included in this repository:
- experimental_script.sh: runs the main script for running EA configurations on problems
- pf_approx_script.sh: calculates Pareto front approximations from archives

In the script files, fill in spaces marked with 'XXXXXX' with your project name. Python has to be set up with venv.
More information on Puhti can be found here: [https://docs.csc.fi/computing/systems-puhti/](https://docs.csc.fi/computing/systems-puhti/)