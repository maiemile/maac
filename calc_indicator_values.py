# Code by @maiemile

# quick setup for the baseline path
BASE_PATH = ''

from desdeo.tools.indicators_unary import hv, igd_plus_indicator, distance_indicators
from pathlib import Path
import numpy as np
import pandas as pd
from multiprocessing import Pool
import utils as util

# all the problems instances
problem_instances = util.get_problem_instances()
algos, cxs, mxs = util.get_all_configuration_options()


def get_experiments() -> list[list]:
    '''
    Fetches the experiments consisting of a problem and a configuration as a list of lists.
    '''

    # Enumerate all possible problem instance + config permutations
    prob_and_config_permutations = [prob + [a, cx, mx] for prob in problem_instances for a in algos for cx in cxs for mx in mxs]
    
    return prob_and_config_permutations


def calculate_indicator_values(config:str, ideal_vector, nadir_vector, pf_approx) -> None:
    '''
    Calculates the performance indicator values of the given configuration on a problem.
    '''

    log_path = Path(BASE_PATH + 'igd_values_log.txt')
    
    # try to fetch the archive, in case it doesn't exist, log an error
    try:
        file_name = Path(BASE_PATH + 'archived_pops/' + config + '.txt')
        archive = np.array(pd.read_table(file_name, sep=" ", header=None))
    except:
        log_text = config + " ERROR"
        prog_log = open(log_path, "a")
        prog_log.write(config + " ERROR" + "\n")
        prog_log.close()
        return

    # archive normalization
    normalized_archive = (archive-ideal_vector) / (nadir_vector-ideal_vector)
    print("archive normalized", config, flush=True)

    # normalize the PF approximation too
    normalized_pf_approx = (pf_approx-ideal_vector) / (nadir_vector-ideal_vector)

    # TOO SLOW FOR PROBLEMS WITH MANY OBJECTIVE FUNCTIONS
    #hypervolume_metric = hv(normalized_archive, 1.001)
    #print("hv calculated", config, flush=True)

    # calculate IGD and IGD+ indicators
    regular_igd = distance_indicators(normalized_archive, normalized_pf_approx)
    igd_plus = igd_plus_indicator(normalized_archive, normalized_pf_approx)
    print("igd calculated", config, flush=True)

    log_text = config + ' ' + str(regular_igd.igd) + ' ' + str(igd_plus.igd_plus)

    prog_log = open(log_path, "a")
    prog_log.write(log_text + "\n")
    prog_log.close()

    return


def calc_ind_val_problem(prob_name:str, n_vars:int, n_objs:int, algo:str, cx:str, mx:str) -> None:
    '''
    Helper function for fetching the Pareto front approximation as well as the ideal and nadir vectors 
    of a given problem. Calls calculate_indicator_values with the given configuration and problem.
    '''
    # create the unique problem name plus configuration
    prob_name_print = prob_name + '-' + str(n_objs) + 'obj'
    file_name = Path(BASE_PATH + 'approx_pfs/' + prob_name_print + '.txt')
    pf_approx = np.array(pd.read_table(file_name, sep=" ", header=None))
    print("PF approx fetched", prob_name_print, flush=True)
    ideal_vector = np.min(pf_approx, axis=0)
    nadir_vector = np.max(pf_approx, axis=0)
    
    configuration = prob_name_print + "-" + algo + "-" + cx + "-" + mx
    calculate_indicator_values(configuration, ideal_vector, nadir_vector, pf_approx)
    
    return


def do() -> None:
    experiment_list = get_experiments()
    not_completed = []
    completed_experiments = []

    # load completed experiments if they exist
    try:
        with open(BASE_PATH + 'igd_values_log.txt', 'r') as file:
            for line in file:
                completed_experiments.append(line)
    except:
        pass

    # find experiments that have already been completed
    for experiment in experiment_list:
        completed = False
        prob_name_print = experiment[0] + "-" + str(experiment[2]) + "obj"
        for line in completed_experiments:     
            split_line = line.split() 
            config = split_line[0].split('-')
            if (config[0]+'-'+config[1] == prob_name_print and config[2] == experiment[3] and 
                config[3] == experiment[4] and config[4] == experiment[5]):
                completed = True
                break
        if completed == False:
            not_completed.append(experiment)

    print(len(experiment_list))
    print(len(not_completed))

    print("started")
    with Pool(processes=20) as pool:
        pool.starmap(calc_ind_val_problem, not_completed)
        pool.terminate()
        pool.join()
