# code by @maiemile


from pathlib import Path
import pandas as pd
from scipy.spatial.distance import cdist
import numpy as np
from multiprocessing import Pool, cpu_count

from desdeo.tools.non_dominated_sorting import non_dominated_merge
import polars as pl
import utils as util
from generate_database import query_data

######################################################

algos, cxs, mxs = util.get_all_configuration_options()
BASE_PATH = util.load_param_config('base_path')

def calc_pf_approx(problem_id:int, pf_approx_size:int=2000) -> None:
    '''
    Calculates the Pareto front approximation for the given problem.
    Uses non-dominated archives of algorithm configuration runs on the given problem.
    Non-dominated archives are combined and the non-dominated solutions on that set are calculated.
    Distance-based subset selection is performed to limit the size of the approximation.
    Results are saved to a CSV file.
    '''
    pf = []
    counter = 0

    # get all run_ids where the current problem was run
    sql = '''SELECT run_id FROM runs WHERE problem_id = ?'''
    runs = query_data(sql, (problem_id,))

    for run in runs:
        run_id = run[0]
        # load the archived non-dominated solutions of the run if they exist
        try:
            path = Path(BASE_PATH + 'archived_pops/' + str(run_id) + '.csv')
            pf2 = np.array(pd.read_csv(path))
        except:
            continue

        try:
        # perform non-dominated merge with the current PF approximation
            mask1, mask2 = non_dominated_merge(pf, pf2)
            df1 = pl.from_numpy(pf)
            df2 = pl.from_numpy(pf2)
            pf = pl.concat([df1.filter(mask1), df2.filter(mask2)])
            pf = np.array(pf)
            counter += 1
            print("NDM done", counter, problem_id)
        # unless there is nothing to merge with, then set the first archive as the initial non-dominated population
        except:
            pf = pf2
            counter += 1

    print('--------------------')
    print(len(pf), problem_id)
    print('--------------------')
    # if PF approximation is too large, perform distance-based subset selection
    if len(pf) > pf_approx_size:
        chosen = [pf[0]]
        for i in range(pf_approx_size-1):
            distances = cdist(pf, chosen, metric='chebyshev').min(axis=1)
            chosen.append(pf[np.argmax(distances)])
            print(problem_id, i)
    # otherwise just use the full PF approximation
    else:
        chosen = pf

    # save the PF approximation to a file
    path = Path(BASE_PATH + 'approx_pfs/' + str(problem_id) + '.csv')
    util.write_to_csv(path, chosen)


def setup_multiprocessing() -> None:
    '''
    A helper method for setting up multiprocessing for Pareto front approximation calculations.
    '''

    # load all problem instances from the database
    sql_query = '''SELECT problem_id from problems'''
    problem_instances = query_data(sql_query)

    # the data format must be fixed for multiprocessing
    fixed_prob_instances = []
    for prob in problem_instances:
        fixed_prob_instances.append(prob[0])

    # TODO: Currently, all PF approximation are calculated every time.
    # Improve the implementation by only calculating approximations for 
    # problems that either don't have one yet or which have had enough runs (say, 1st seed of all configurations)

    # spread the calculations across the CPUs with multiprocessing
    with Pool(processes=cpu_count()) as pool:
        pool.map(calc_pf_approx, fixed_prob_instances)
        pool.terminate()
        pool.join()


def do() -> None:
    setup_multiprocessing()


if __name__ == "__main__":
    problem_instances = util.get_problem_instances()
    
    probs = []
    
    for prob in problem_instances:
        probs.append(prob[0]+'-'+str(prob[2])+'obj')

    calc_pf_approx(probs)
