# code by @maiemile

from desdeo.emo import (
    algorithms,termination,generator,repair
)
import numpy as np
from pathlib import Path
import os.path
from multiprocessing import Pool, cpu_count, set_start_method
from desdeo.emo.options.generator import ArchiveGeneratorOptions

from generate_database import query_data
import utils as util
from sampling import ela_features
import polars as pl

pop_sizes = util.get_default_pop_sizes()

# Load the filename of the database and the base path
database = util.load_param_config('database_file')
BASE_PATH = str(util.load_param_config('base_path'))


def run_experiment(run_id:int, ea_id:int, problem_id:int, seed:int, target_evals:int, options:dict) -> None:
    '''
    Run the experiment defined by run_id. Creates the EA template based on the ea_id, seed and target_evaluations.
    Loads the problem defined by problem_id. Runs the EA configuration on that problem and stores the non-dominated
    archive and final population in CSV files.
    '''

    print(f"run ID: {run_id}, EA ID: {ea_id}, problem ID: {problem_id}, seed: {seed}, target evaluations: {target_evals}")

    # prepare SQL statements to fetch data on the EA configuration and problem
    sql_statement = '''SELECT selection,crossover,mutation FROM eas WHERE ea_id = ?'''
    ea_data = query_data(sql_statement, (ea_id,))

    sql_statement_prob = '''SELECT name,obj,var FROM problems WHERE problem_id = ?'''
    prob_data = query_data(sql_statement_prob, (problem_id,))

    # unload the problem data
    prob_name, n_obj, n_var = prob_data
    pop_size = pop_sizes[n_obj]

    # fetch the corresponding operators
    main_template = options["selection"][1][ea_data[0]]
    main_template.template.crossover = options["crossover"][1][ea_data[1]]
    main_template.template.mutation = options["mutation"][1][ea_data[2]]

    # max_generations must be set correctly for NUM
    if ea_data[2] == "NUM":
        main_template.template.mutation.max_generations = int(target_evals/pop_size) 
    
    # fix the termination, generator and repair
    main_template.template.termination = termination.MaxEvaluationsTerminatorOptions(max_evaluations=target_evals)
    if seed == 1: # ELA features only need to be calculated on the first seed 
        # calculate the ELA features where the sample is the initial population
        aggregators = util.get_default_aggregators()
        initial_pop, outputs = ela_features((problem_id,prob_name,n_obj,n_var), aggregators, sample_size=pop_size)
        main_template.template.generator = ArchiveGeneratorOptions(solutions=pl.DataFrame(initial_pop, schema=[f"x_{i}" for i in range(1, n_var+1)])
                                                                   , outputs=pl.DataFrame(outputs, schema=[f"f_{i}" for i in range(1, n_obj+1)]))
    else:
        main_template.template.generator = generator.LHSGeneratorOptions(n_points = pop_size)
    main_template.template.repair = repair.ClipRepairOptions()
    main_template.template.seed = seed

    # fix the population size for different operators
    try:
        main_template.template.selection.reference_vector_options.number_of_vectors = pop_size
    except:
        pass

    try:
        main_template.template.selection.population_size = pop_size
    except:
        pass

    try: 
        main_template.template.mate_selection.winner_size = pop_size
    except:
        pass

    problem = util.get_problem_object(prob_name, n_obj, n_var)

    # construct the EA configuration and problem
    solver, extras = algorithms.emo_constructor(emo_options=main_template, problem=problem)
    res = solver()

    # fetch the final population and the archive
    objective_names = [obj.name for obj in problem.objectives]
    final_pop = np.array(res.optimal_outputs[objective_names])
    archived_solutions = np.array(extras.archive.results.optimal_outputs[objective_names])
    #print(f"Total number of non-dominated solutions in archive: {len(extras.archive.results.optimal_outputs)}")

    # save the archived solutions and the final population to csv files identified by the run ID
    util.write_to_csv(Path(BASE_PATH + 'archived_pops/' + str(run_id) + '.csv'), archived_solutions)
    util.write_to_csv(Path(BASE_PATH + 'archived_final_pops/' + str(run_id) + '.csv'), final_pop)


def do(setup:util.ExperimentalSetup):
    '''
    The method for setting up the experiments.
    The runs are fetched from the 'runs' table in the database.
    This function automatically checks which runs have been completed and 
    only runs uncompleted ones.
    Multiprocessing is utilized according to the number of CPUs available.
    '''

    # Configuration options
    options = setup.options

    # get the full list of experiments from the table "runs"
    sql_fetch_runs = '''SELECT run_id, ea_id, problem_id, seed, target_evals FROM runs'''
    data = query_data(sql_fetch_runs)
    # (run_id, ea_id, problem_id, seed, target_evals)

    # find uncompleted runs by identifying if archives have been saved for them
    uncompleted_runs = []
    for row in data:
        if not os.path.isfile(Path(BASE_PATH + 'archived_pops/' + str(row[0]) + '.csv')):
            new_row = list(row)
            # TODO: currently we just add the options here, in the future it might make sense to load 
            # them in the run_experiment function using dedicated JSONs and problem info.
            new_row.append(options)
            uncompleted_runs.append(new_row)

    # sort in ascending order based on the seed
    uncompleted_runs.sort(key=lambda tup: tup[3])

    set_start_method('spawn')
    # Create a pool of workers and finish the uncompleted runs
    with Pool(processes=cpu_count()) as pool:
        # chunksize=1 for making sure that the calculation of the first seeds begins first  
        pool.starmap(run_experiment, uncompleted_runs, chunksize=1) 
        pool.terminate()
        pool.join()
    

