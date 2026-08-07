# code by @maiemile

import os
from pathlib import Path
import pandas as pd
from scipy.spatial.distance import cdist
import numpy as np
from multiprocessing import Pool, cpu_count

from desdeo.tools.non_dominated_sorting import non_dominated_merge
import polars as pl
import utils as util
from generate_database import query_data
from datetime import datetime

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename='maac_pf.log', level=logging.INFO)

######################################################

BASE_PATH = util.load_param_config('base_path')

def calc_pf_approx(problem_id:int, pf_approx_size:int=None) -> None:
    '''
    Calculates the Pareto front approximation for the given problem.
    Uses non-dominated archives of algorithm configuration runs on the given problem.
    Non-dominated archives are combined and the non-dominated solutions on that set are calculated.
    Distance-based subset selection is performed to limit the size of the approximation.
    Results are saved to a CSV file.
    '''
    
    # get all run_ids where the current problem was run
    sql = '''SELECT run_id FROM runs WHERE problem_id = ?'''
    runs = query_data(sql, (problem_id,))

    counter = 0

    # if reference PF size has not been set, use the default value
    # for the number of objective functions this problem has
    if pf_approx_size == None:
        sql_prob = '''SELECT obj FROM problems WHERE problem_id = ?'''
        n_obj = query_data(sql_prob, (problem_id,))[0]
        pf_approx_size = util.get_default_ref_pf_size(n_obj)

    pf = []
    for run in runs:
        run_id = run[0]
        # load the archived non-dominated solutions of the run if they exist
        try:
            path = Path(BASE_PATH + 'archives_temp/' + str(run_id) + '.csv')
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
            if counter % 25 == 0:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                logger.info(f"{timestamp} | Problem {problem_id} at archive {counter}/{len(runs)}")
        # unless there is nothing to merge with, then set the first archive as the initial non-dominated population
        except:
            pf = pf2

    # save the temporary PF approximation to a file
    path = Path(BASE_PATH + 'approx_pfs_temp/' + str(problem_id) + '.csv')
    util.write_to_csv(path, np.array(pf))

    logger.info(f"Problem {problem_id} has a reference PF of size {len(pf)}")
    # if PF approximation is too large, perform distance-based subset selection
    if len(pf) > pf_approx_size:
        chosen = [pf[0]]
        for i in range(pf_approx_size-1):
            distances = cdist(pf, chosen, metric='chebyshev').min(axis=1)
            chosen.append(pf[np.argmax(distances)])
            if i % 50 == 0:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                logger.info(f"{timestamp} | Problem {problem_id} at size {i}/{pf_approx_size}")
    # otherwise just use the full PF approximation
    else:
        chosen = pf

    # save the PF approximation to a file
    path = Path(BASE_PATH + 'approx_pfs/' + str(problem_id) + '.csv')
    util.write_to_csv(path, chosen)
    logger.info(f"PF of problem {problem_id} saved")


def do() -> None:
    '''
    A helper method for setting up multiprocessing for Pareto front approximation calculations.
    '''

    # load all problem instances from the database
    sql_query = '''SELECT problem_id from problems'''
    problem_instances = query_data(sql_query)

    # the data format must be fixed for multiprocessing
    fixed_prob_instances = []
    for prob in problem_instances:
        # don't calculate the PF approximation for problems that already have one
        if not os.path.isfile(Path(BASE_PATH + 'approx_pfs/' + str(prob[0]) + '.csv')):
            fixed_prob_instances.append(prob[0])

    # spread the calculations across the CPUs with multiprocessing
    with Pool(processes=cpu_count()) as pool:
        pool.map(calc_pf_approx, fixed_prob_instances)
        pool.terminate()
        pool.join()


if __name__ == "__main__":
    do()
