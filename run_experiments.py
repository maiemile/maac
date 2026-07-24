# code by @maiemile

from desdeo.emo import (
    algorithms,mutation, crossover,termination,generator,repair
)
import numpy as np
from pathlib import Path
import os.path
import multiprocessing as mp
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

    # Create a pool of workers and finish the uncompleted runs
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # chunksize=1 for making sure that the calculation of the first seeds begins first  
        pool.starmap(run_experiment, uncompleted_runs, chunksize=1) 
        pool.terminate()
        pool.join()


if __name__ == "__main__":
    mp.set_start_method('spawn')
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
    do(setup)

