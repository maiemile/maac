# Code by @maiemile

import matplotlib.pyplot as plt


def load_files_config() -> bool:
    '''
    Reads the config.txt file and returns the value of the "load_models" variable.
    '''
    import configparser
    from pathlib import Path

    # use the config file to determine whether to load configurator models from files or training new ones
    config_parser = configparser.RawConfigParser()   
    config_path = Path("config.txt")
    config_parser.read(config_path)
    load_from_files = config_parser.get('general', 'load_models')

    return load_from_files


def create_igd_array_and_dict(file_name):
    igd_dict = {}
    igd_array = []
    igd_array_regr = []

    # Create a dictionary with the problem+config string as a key and igd and igd+ as the value
    with open(file_name, 'r') as file:
        for line in file:
            split_line = line.split() 
            problem_and_config = split_line[0]
            try:
                igd = float(split_line[1])
                igd_plus = float(split_line[2])
            except:
                continue # Error
            igd_dict[problem_and_config] = (igd,igd_plus)
            split_config = split_line[0].split('-')
            problem = split_config[0] + '-' + split_config[1]
            config = split_config[2] + '-' + split_config[3] + '-' + split_config[4]
            igd_array.append([problem, config, igd, igd_plus])
            igd_array_regr.append([problem, split_config[2], split_config[3], split_config[4], igd, igd_plus])

    return igd_array, igd_dict, igd_array_regr


def get_problem_instances() -> list[list]:
    '''
    Returns a list of every individual problem instance.
    Each problem instance corresponds to one list within the list and these lists contain
    the name of the problem, the number of decision variables and the number of objective functions, in this order. 
    '''
    # all the problems instances
    problem_instances = [
        # DTLZ problems (3, 4, 6 and 9 objectives, 4 * 7 = 28 instances)
        ["dtlz1", 7, 3],["dtlz2", 10, 3],["dtlz3", 10, 3],["dtlz4", 10, 3],["dtlz5", 10, 3],["dtlz6", 10, 3],["dtlz7", 10, 3],
        ["dtlz1", 11, 4],["dtlz2", 15, 4],["dtlz3", 15, 4],["dtlz4", 15, 4],["dtlz5", 15, 4],["dtlz6", 15, 4],["dtlz7", 15, 4], #DTLZ5-7 with 4 objectives doesn't work, "not implemented yet" in pymoo
        ["dtlz1", 11, 6],["dtlz2", 15, 6],["dtlz3", 15, 6],["dtlz4", 15, 6],["dtlz5", 15, 6],["dtlz6", 15, 6],["dtlz7", 15, 6],
        ["dtlz1", 11, 9],["dtlz2", 15, 9],["dtlz3", 15, 9],["dtlz4", 15, 9],["dtlz5", 15, 9],["dtlz6", 15, 9],["dtlz7", 15, 9],
        # WFG problems (3, 4, 6 and 9 objectives, 4 * 9 = 36 instances)
        ["wfg1", 10, 3],["wfg2", 10, 3],["wfg3", 10, 3],["wfg4", 10, 3],["wfg5", 10, 3],["wfg6", 10, 3],["wfg7", 10, 3],["wfg8", 10, 3],["wfg9", 10, 3],
        ["wfg1", 15, 4],["wfg2", 14, 4], ["wfg3", 14, 4],["wfg4", 15, 4],["wfg5", 15, 4],["wfg6", 15, 4],["wfg7", 15, 4],["wfg8", 15, 4],["wfg9", 15, 4], #WFG2 and WFG3 require an even number of decision variables
        ["wfg1", 15, 6],["wfg2", 14, 6], ["wfg3", 14, 6],["wfg4", 15, 6],["wfg5", 15, 6],["wfg6", 15, 6],["wfg7", 15, 6],["wfg8", 15, 6],["wfg9", 15, 6],
        ["wfg1", 18, 9],["wfg2", 18, 9], ["wfg3", 18, 9],["wfg4", 18, 9],["wfg5", 18, 9],["wfg6", 18, 9],["wfg7", 18, 9],["wfg8", 18, 9],["wfg9", 18, 9],
        # RE problems, 8 instances
        ["re31", 3, 3],["re32", 4, 3],["re33", 4, 3],["re34", 5, 3],["re37", 4, 3],
        #["re41", 7, 4], # not included due to unexpected results while calculating the approximate Pareto front
        #["re42", 6, 4],["re61", 3, 6], 
    ]
    return problem_instances


def get_test_problems() -> list[str]:
    '''
    Returns a list of all test problems (only string format)
    '''
    test_problems = ['dtlz1-4obj', 'dtlz2-3obj', 'dtlz3-9obj', 'dtlz5-6obj', 'dtlz7-9obj', 
                         'wfg1-4obj', 'wfg3-3obj', 'wfg4-9obj', 'wfg6-6obj', 'wfg8-9obj'
                         , 're31-3obj', 're32-3obj', 're33-3obj', 're34-3obj', 're37-3obj']
    return test_problems


def get_all_configuration_options() -> list[list]:
    '''
    Returns a dictionary with each key corresponding to a configuration parameter and the value to the list of values for that parameter

    '''
    algos = ["nsga3", "rvea", "ibea"]
    cxs = ["SBX", "Balpha", "Single", "Local"]
    mxs = ["BPM", "MPTM", "NUM", "PM"]

    return algos, cxs, mxs


def get_all_configurations() -> list[str]:
    '''
    Returns a list of all configurations used in the following string format "algorithm-crossover-mutation"
    
    :return: All configurations in the following format: "algorithm-crossover-mutation"
    :rtype: list[str]
    '''
    algos, cxs, mxs = get_all_configuration_options()

    # rvea-NUM is ignored due to too many errors
    configs = [a+"-"+cx+"-"+mx for a in algos for cx in cxs for mx in mxs if a != 'rvea' or mx != 'NUM']

    return configs


# TODO: check the folder indicator_data, and find all suffixes
def get_default_aggregators() -> list[str]:
    '''
    Returns the default list of aggregators used for exploratory landscape analysis (ELA)
    '''

    return ["max", "min", "avg", "sd", "nds", "moo"]


def get_benchmark_configurations() -> list[str]:
    '''
    Returns the default list of benchmark configurations used for performance profile plots or other comparisons against the configurators.
    '''

    return ['ibea-SBX-NUM', 'nsga3-SBX-BPM']



def get_default_pop_sizes() -> dict:
    '''
    Obtain a dictionary of the default population sizes based on the number of objective functions.
    Calculated using a simplex-lattice formula.
    '''
    return {3: 105, 4: 120, 6: 132, 9: 210} # from the RVEA article, partially interpolated


def get_labels_from_file(labels: list[str], feat_sets: list[str]) -> list[str]:
    '''
    Obtain all ELA feature names from files corresponding to the feature set names.
    Returns a list of strings.
    '''
    import os
    for f_set in feat_sets:
        filename = None
        for x in os.listdir('ela_features'):
            # TODO: could allow other file types
            if x.endswith(".txt") and f_set in x:
                filename = x
        
        # If no classifier models were found, raise an exception
        if filename == None:
            raise Exception(f'No ELA files found for aggregator {f_set}. Feature names could not be loaded.')
        
        with open(f'ela_features\\{filename}', 'r') as file:
            for line in file:
                split_line = line.split() 

                # runtime isn't a proper feature
                if "costs_runtime" in split_line[0]:
                    continue
                labels.append(split_line[0]+f'_{f_set.upper()}')

    return labels


def load_data(data_array, feat_sets, problem_instances):
    data = []
    for res in data_array:
        prob = res[0]
        split_prob_name = prob.split('-')
        name = split_prob_name[0]
        objectives = int(split_prob_name[1][0])
        variables = 0
        for i in range(len(problem_instances)):
            # TODO: this needs to be changed to handle same problem instances with varying number of decision variables 
            if problem_instances[i][0] == name and problem_instances[i][2] == objectives:
                variables = problem_instances[i][1]

        res = res + [objectives, variables]

        for f_set in feat_sets:
            # Load the ELA features values 
            ela_feat = []
            with open(f'ela_features\\{prob}_{f_set}.txt', 'r') as file:
                for line in file:
                    split_line = line.split() 
                    if "costs_runtime" in split_line[0]:
                        continue
                    ela_feat.append(float(split_line[1]))
            res = res + ela_feat
        data.append(res)

    return data


def save_and_print_results(result_dicts, result_dict_names, result_folder=None):
    from pathlib import Path

    for i in range(len(result_dicts)):
        res_dict = result_dicts[i]
        print(result_dict_names[i] + "\n")  

        sorted_res_dict = sorted(res_dict.items(), key=lambda kv: kv[1])    

        for values in sorted_res_dict:
            print(values[0], values[1]) 

        print("\n" + "="*30 +"\n")  

        if result_folder != None:
            # save the sorted results to text files
            path = Path(f'{result_folder}{result_dict_names[i]}.txt')
            with open(path, "w") as file:
                for line in sorted_res_dict:
                    file.write(" ".join(str(item) for item in line) + "\n")


def get_dataframe_for_performance_profile(igd_dict, configs, problems, igd_values=None, config_labels=['configurator']):
    import pandas as pd
    import numpy as np

    if igd_values != None:
        try:
            if len(igd_values[0]) != None:
                full_igd_values = igd_values
        except:
            full_igd_values = [igd_values]
    else:
        full_igd_values = []

    #print(full_igd_values)
    if configs != None:
        for config in configs:
            igd_values_c = []
            for problem in problems:
                problem_plus_config = problem + '-' + config
                igd_values_c.append(igd_dict[problem_plus_config][0])
            full_igd_values.append(igd_values_c)

        #print(full_igd_values)
        labels = config_labels + configs
    else:
        labels = config_labels

    igd_df = pd.DataFrame(np.array(full_igd_values).T, columns=labels)

    return igd_df


def create_performance_profile_plot(igd_dict: dict, igd_values: list[float], configs: list[str], test_problems: list[str], fig_name:str="img",
                                    config_labels = ['configurator'], font_size = 10) -> None:
    """
    Creates a performance profile plot with the given data.
    
    :param igd_dict: Dictionary of IGD values achieved by all configs listed in 'configs' for all problems listed in 'test_problems'
    :param igd_values: IGD values the configurator model achieved
    :param configs: Configurations to plot in the graph
    :param test_problems: A list of problems used
    :param fig_name: Name of the created figure
    :param config_labels: Additional labels of configuration in addition to "configs"
    :param font_size: Controls the font size of the legend
    """

    import perfprof

    data_array = get_dataframe_for_performance_profile(igd_dict, configs, test_problems, igd_values, config_labels).to_numpy()

    palette = ['o-C0', 'o:C1', 'o--C2', 'o-.C3', 'o-C4', 'o:C5', 'o-C6', 'o--C0', 'o:C0', 'o-C1', 'o--C1', 'o:C2', 'o-C2',
               'o:C3', 'o--C3', 'o-C3', 'o-.C4', 'o--C4', 'o:C4', 'o-.C5', 'o-C5', 'o--C5', 'o-.C6', 'o--C6', 'o:C6',
               'o-.C7', 'o-C7', 'o--C7', 'o:C7', 'o-.C8', 'o-C8', 'o--C8', 'o:C8', 'o-.C9', 'o-C9', 'o--C9', 'o:C9',
               'v:C3', 'v--C3', 'v-C3', 'v-.C4', 'v--C4', 'v:C4', 'v-.C5', 'v-C5', 'v--C5', 'v-.C6', 'v--C6', 'v:C6',
               'v-.C7', 'v-C7', 'v--C7', 'v:C7', 'v-.C8', 'v-C8', 'v--C8', 'v:C8', 'v-.C9', 'v-C9', 'v--C9', 'v:C9']

    perfprof.perfprof(data_array, linestyle=palette, thmax=5., markersize=4, markevery=[0])

    if configs != None:
        configs_uc = []
        for config in configs:
            configs_uc.append(config.upper().replace("NSGA3", "NSGA-III"))
        configs_labels = config_labels + configs_uc
    else:
        configs_labels = config_labels
    plt.legend(configs_labels, loc=4, fontsize=font_size)
    plt.savefig(f'figures\\perf_prof\\{fig_name}.pdf')
    plt.show()
