# code by @maiemile

# quick setup for the baseline path
BASE_PATH = ''

from pathlib import Path
from scipy.spatial.distance import cdist

import numpy as np

from pathlib import Path

import pandas as pd

from desdeo.tools.non_dominated_sorting import non_dominated_merge
import polars as pl
from multiprocessing import Pool
import utils as util

######################################################

algos = ["nsga3", "rvea", "ibea"]
cxs = ["SBX", "Balpha", "Single", "Local"]
mxs = ["BPM", "MPTM", "NUM", "PM"]

def calc_pf_approx(prob):
    prob_name_print = prob
    pf = []
    counter = 0
    for cx in cxs:
        for mx in mxs:
            for algo in algos:      
                configuration = "-" + algo + "-" + cx + "-" + mx
                try:
                    path = Path(BASE_PATH + '/archived_pops_v3/' + prob_name_print + configuration + '.txt')
                    pf2 = np.array(pd.read_table(path, sep=" ", header=None))
                except:
                    continue
                
                try:
                    mask1, mask2 = non_dominated_merge(pf, pf2)
                    df1 = pl.from_numpy(pf)
                    df2 = pl.from_numpy(pf2)
                    pf = pl.concat([df1.filter(mask1), df2.filter(mask2)])
                    pf = np.array(pf)
                    counter += 1
                    print("NDM done", counter, prob_name_print + configuration)
                except:
                    pf = pf2
                    counter += 1
    
    print('--------------------')
    print(len(pf), prob_name_print)
    print('--------------------')
    if len(pf) > 2000:
        chosen = [pf[0]]
        for i in range(1999):
            distances = cdist(pf, chosen, metric='chebyshev').min(axis=1)
            chosen.append(pf[np.argmax(distances)])
            print(prob_name_print, i)
    else:
        chosen = pf
    path = Path(BASE_PATH + '/approx_pfs_v3/' + prob_name_print + '.txt')
    with open(path, "w") as file:
        for line in chosen:
            file.write(" ".join(str(x) for x in line.tolist()) + "\n")

if __name__ == "__main__":
    print("starting")
    problem_instances = util.get_problem_instances()
    
    probs = []
    
    for prob in problem_instances:
        probs.append(prob[0]+'-'+str(prob[2])+'obj')

    with Pool(processes=20) as pool:
        pool.map(calc_pf_approx, probs)
        pool.terminate()
        pool.join()
        
