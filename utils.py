# Code by @maiemile

import math
import matplotlib.pyplot as plt
import configparser
from pathlib import Path
import numpy as np
import perfprof
import reprob
import csv
from pathlib import Path
from desdeo.problem.testproblems import re_problem as re
from desdeo.problem import Problem
from desdeo.problem.external import pymoo_provider
from sklearn.metrics import (confusion_matrix ,ConfusionMatrixDisplay)


class ExperimentalSetup():
    '''
    Class for defining the parameters used for the experiments.
    '''

    def __init__(self, options:dict, problems:list[list]):
        self.options = options
        self.problems = problems


def write_to_csv(file:Path, data:np.array) -> None:
    '''
    Writes a CSV file at the given file using the data.
    '''
    with open(file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data) 


def load_param_config(param:str) -> bool | str:
    '''
    Reads the config.txt file and returns the value of requested parameter.
    '''

    # use the config file to load the desired parameter value
    config_parser = configparser.RawConfigParser()   
    config_path = Path("config.txt")
    config_parser.read(config_path)
    param_value = config_parser.get('general', param)
    if param_value == 'None':
        return ""
    if param_value == 'True':
        return True
    if param_value == 'False':
        return False
    return param_value


def convert_data(df):
    '''
    Converts binary variables to integers.
    '''
    to_convert = list(df.select_dtypes(exclude=["float64", "int"]).columns)
    for col in to_convert:
        df[col] = df[col].apply(
            lambda x: float(int.from_bytes(x, byteorder="little"))
    )
    return df


def get_param_names() -> list[str]:
    '''
    Returns the EA configuration parameter names.
    '''
    from generate_database import query_data
    # find the EA parameter names
    sql ='''SELECT name FROM PRAGMA_TABLE_INFO('eas') WHERE name!='ea_id' AND name!='name' '''
    res = query_data(sql)
    params = []
    for item in res:
        params.append(item[0])

    return params


def determine_single_best_solver(test_problems:list[int]=[], indicator:str = None) -> int:
    '''
    Returns the ID of the single best solver, ignoring the supplied test problems.
    '''
    from generate_database import query_data, get_median_by_config_and_problem

    if indicator == None:
        # Load the default indicator
        indicator = load_param_config('indicator')

    # load all EA ids
    sql = '''SELECT ea_id FROM eas'''
    res = query_data(sql)

    ea_ids = []
    for row in res:
        ea_ids.append(row[0])

    # load all problem ids
    sql = '''SELECT problem_id FROM problems'''
    res = query_data(sql)

    prob_ids = []
    for row in res:
        prob_ids.append(row[0])

    # loop through all EAs and problems
    # get the median of that combination
    # then calculate the mean of the medians across all problems
    avgs = {}
    for ea_id in ea_ids:
        median_list = []
        for prob_id in prob_ids:
            if prob_id in test_problems:
                continue
            median = get_median_by_config_and_problem(prob_id, ea_id, indicator)
            # if the median could not be calculated, replace it with a large value
            if median == None:
                median_list.append(99999999)
            else:
                median_list.append(median)

        average = np.mean(median_list)
        avgs[ea_id] = average

    #for k,v in avgs.items():
    #    print(k,v)

    # the SBS is the configuration with the smallest average median 
    sbs = min(avgs, key=avgs.get)

    return sbs


def create_confusion_matrices(y_test, y_pred_test, model_name:str, titles:list[str]=None) -> None:
    '''
    Creates a separate confusion matrix for each output. 
    The confusion matrices are saved in a folder.
    '''
    # Confusion matrix for each output
    params = len(y_test[0])
    fig, axes = plt.subplots(math.ceil(params/3), 3, figsize=(12,4*math.ceil(params/3)))
    for j in range(params):
        labels = np.unique([y_test[:,j], y_pred_test[:,j]])
        cm = confusion_matrix(y_test[:, j], y_pred_test[:, j], labels=labels)
        print(cm)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=axes[j], colorbar=False)

    if titles == None:
        titles = get_param_names()

    # only set the x and y labels to the left and below the full plot
    for i in range(params):
        axes[i].set_title(titles[i])
        if i % 3 != 0:
            axes[i].set_ylabel('')
        if i < params-3:
            axes[i].set_xlabel('')

    plt.savefig(f'figures\\confusion_matrices\\{model_name}.pdf')
    plt.show()


def get_re_problems() -> dict:
    '''
    Returns all RE problems in a dictionary. Each value is the function name of the problem in DESDEO.
    '''
    return {"re31": reprob.re31, "re32": reprob.re32, "re33": reprob.re33, "re34": re.re34, "re37": re.re37, 
            "re41": re.re41, "re42": reprob.re42, "re61": re.re61}


def get_problem_object(prob_name:str, n_obj:int, n_var:int) -> Problem:
    '''
    Returns the Problem object with the given information. Works for DTLZ, RE and WFG problems.
    '''
    if prob_name[:2] == 're':
        problem_obj = get_re_problems()[prob_name]()
    elif prob_name[:3] == 'wfg' or prob_name[:4] == 'dtlz':
        problem_obj = pymoo_provider.create_pymoo_problem(pymoo_provider.PymooProblemParams(name=prob_name, n_var=n_var, n_obj=n_obj))
    else:
        raise Exception('Invalid problem name.')

    return problem_obj


def get_test_problems(test_problems:list[str | int]=[]) -> list[int]:
    '''
    Returns a list of all test problems (only string format)
    '''
    from generate_database import query_data

    test_problems_fixed = []
    for problem in test_problems:
        # problems given in string or integer format must be handled differently
        if isinstance(problem, str):
            sql = '''SELECT problem_id FROM problems WHERE name = ?'''
            res = query_data(sql, (problem,))
            if len(res) == 1:
                test_problems_fixed.append(res[0])
            elif len(res) > 1:
                for prob in res:
                    test_problems_fixed.append(prob[0])
        elif isinstance(problem, int):
            test_problems_fixed.append(problem)
        else:
            pass

    # remove duplicates and sort the list
    test_probs = list(set(test_problems_fixed))
    test_probs.sort()

    return test_probs


def get_default_aggregators() -> list[str]:
    '''
    Returns the default list of aggregators used for exploratory landscape analysis (ELA)
    '''

    return ["max", "min", "avg", "sd", "nds", "moo"]


def get_default_pop_sizes() -> dict:
    '''
    Obtain a dictionary of the default population sizes based on the number of objective functions.
    Calculated using a simplex-lattice formula.
    '''
    return {3: 105, 4: 120, 6: 132, 9: 210} # from the RVEA article, partially interpolated


def get_default_ref_pf_size(n_obj:int) -> int:
    '''
    Returns the default reference Pareto front size for the given number of objective functions.
    '''
    pf_sizes = {3: 1500, 4: 2000, 6: 3000, 9: 4500}
    return pf_sizes[n_obj]


def save_and_print_results(result_dicts: list[dict], result_dict_names:list[str], result_folder:str=None) -> None:
    '''
    Saves results from dictionaries in files corresponding to their names.
    '''

    for i in range(len(result_dicts)):
        res_dict = result_dicts[i]
        print(result_dict_names[i] + "\n")  

        # sort according to the values in the dictionary
        sorted_res_dict = sorted(res_dict.items(), key=lambda kv: kv[1])    

        for values in sorted_res_dict:
            print(values[0], values[1]) 

        print("\n" + "="*30 +"\n")  

        # only save the results if the result folder has been set
        if result_folder != None:
            # save the sorted results to text files
            path = Path(f'{result_folder}{result_dict_names[i]}.txt')
            with open(path, "w") as file:
                for line in sorted_res_dict:
                    file.write(" ".join(str(item) for item in line) + "\n")


def create_performance_profile_plot(igd_df, configs: list[str], fig_name:str="img", font_size:int = 10) -> None:
    """
    Creates a performance profile plot with the given data.
    
    :param igd_df: A dataframe with the indicator values by configuration: one row per problem, one column per configuration
    :param configs: Configurations to plot in the graph
    :param fig_name: Name of the created figure
    :param font_size: Controls the font size of the legend
    """

    palette = ['o-C0', 'o:C1', 'o--C2', 'o-.C3', 'o-C4', 'o:C5', 'o-C6', 'o--C0', 'o:C0', 'o-C1', 'o--C1', 'o:C2', 'o-C2',
               'o:C3', 'o--C3', 'o-C3', 'o-.C4', 'o--C4', 'o:C4', 'o-.C5', 'o-C5', 'o--C5', 'o-.C6', 'o--C6', 'o:C6',
               'o-.C7', 'o-C7', 'o--C7', 'o:C7', 'o-.C8', 'o-C8', 'o--C8', 'o:C8', 'o-.C9', 'o-C9', 'o--C9', 'o:C9',
               'v:C3', 'v--C3', 'v-C3', 'v-.C4', 'v--C4', 'v:C4', 'v-.C5', 'v-C5', 'v--C5', 'v-.C6', 'v--C6', 'v:C6',
               'v-.C7', 'v-C7', 'v--C7', 'v:C7', 'v-.C8', 'v-C8', 'v--C8', 'v:C8', 'v-.C9', 'v-C9', 'v--C9', 'v:C9']

    # create the performance profile plots
    perfprof.perfprof(igd_df, linestyle=palette, thmax=5., markersize=4, markevery=[0])

    # convert the benchmark configuration names to upper case
    plt.legend(configs, loc=4, fontsize=font_size)
    plt.savefig(f'figures\\perf_prof\\{fig_name}.pdf')
    plt.show()
