# code by @maiemile


from pathlib import Path
from scipy.spatial.distance import cdist

import numpy as np
from pathlib import Path
import pandas as pd

from desdeo.tools.non_dominated_sorting import non_dominated_merge
import polars as pl
import utils as util

######################################################

algos, cxs, mxs = util.get_all_configuration_options()
BASE_PATH = util.load_param_config('base_path')

def calc_pf_approx(prob:str, pf_approx_size:int=2000) -> None:
    '''
    Calculates the Pareto front approximation for the given problem.
    Uses non-dominated archives of algorithm configuration runs on the given problem.
    Non-dominated archives are combined and the non-dominated solutions on that set are calculated.
    Distance-based subset selection is performed to limit the size of the approximation.
    '''
    prob_name_print = prob
    pf = []
    counter = 0

    # loop through all configurations
    for cx in cxs:
        for mx in mxs:
            for algo in algos:      
                configuration = "-" + algo + "-" + cx + "-" + mx

                # load the archived non-dominated solutions if they exist
                try:
                    path = Path(BASE_PATH + 'archived_pops/' + prob_name_print + configuration + '.txt')
                    pf2 = np.array(pd.read_table(path, sep=" ", header=None))
                except:
                    continue
                
                # perform non-dominated merge with the current PF approximation
                try:
                    mask1, mask2 = non_dominated_merge(pf, pf2)
                    df1 = pl.from_numpy(pf)
                    df2 = pl.from_numpy(pf2)
                    pf = pl.concat([df1.filter(mask1), df2.filter(mask2)])
                    pf = np.array(pf)
                    counter += 1
                    print("NDM done", counter, prob_name_print + configuration)
                # unless there is nothing to merge with, then set the first archive as the initial non-dominated population
                except:
                    pf = pf2
                    counter += 1
    
    print('--------------------')
    print(len(pf), prob_name_print)
    print('--------------------')
    # if PF approximation is too large, perform distance-based subset selection
    if len(pf) > pf_approx_size:
        chosen = [pf[0]]
        for i in range(pf_approx_size-1):
            distances = cdist(pf, chosen, metric='chebyshev').min(axis=1)
            chosen.append(pf[np.argmax(distances)])
            print(prob_name_print, i)
    # otherwise just use the full PF approximation
    else:
        chosen = pf

    # save the PF approximation to a file
    path = Path(BASE_PATH + 'approx_pfs/' + prob_name_print + '.txt')
    with open(path, "w") as file:
        for line in chosen:
            file.write(" ".join(str(x) for x in line.tolist()) + "\n")


def setup_multiprocessing() -> None:
    '''
    A helper method for setting up multiprocessing for Pareto front approximation calculations.
    '''
    from multiprocessing import Pool

    print("starting")
    problem_instances = util.get_problem_instances()
    
    probs = []
    
    # format the problems
    for prob in problem_instances:
        probs.append(prob[0]+'-'+str(prob[2])+'obj')

    with Pool(processes=20) as pool:
        pool.map(calc_pf_approx, probs)
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
