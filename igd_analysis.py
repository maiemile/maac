# Code by @maiemile

import statistics
import numpy as np
import utils as util

# igd_data is a dictionary of dictionaries: 
# each problem is a key in the dictionary
# the value of which is a dictionary of
# configuration: igd/igd+ values


def load_data(file_name:str, index_of_data:int) -> tuple[dict, list[dict]]:
    '''
    Loads dictionaries of indicator data.
    One dictionary with a key for each configuration.
    Individual dictionaries for the algorithm, crossover and mutation as keys.
    '''
    igd_data = {}
    igd_data_by_algo = {}
    igd_data_by_crossover = {}
    igd_data_by_mutation = {}

    igd_data_dicts = [igd_data_by_algo, igd_data_by_crossover, igd_data_by_mutation]

    with open(file_name, 'r') as file:
        for line in file:
            split_line = line.split() 
            config = split_line[0].split('-')
            problem = config[0] + '-' + config[1]
            configuration = config[2] + '-' + config[3] + '-' + config[4]

            # ignore the troublesome problems
            if problem in ["re42-4obj", "re61-6obj"]:
                continue
            try:
                igd = float(split_line[index_of_data])
            except:
                continue # Error with IGD+ value

            # this stores data by full configuration
            try:
                igd_data[problem][configuration] = igd
            except:
                igd_data[problem] = {}
                igd_data[problem][configuration] = igd

            # It ain't pretty but it works...
            # stores data by algorithm, crossover, mutation
            keys = [config[2], config[3], config[4]]
            for i in range(len(igd_data_dicts)):
                try:
                    igd_data_dicts[i][problem][keys[i]].append(igd)
                except:
                    try:
                        igd_data_dicts[i][problem][keys[i]] = [igd]
                    except:
                        igd_data_dicts[i][problem] = {}
                        igd_data_dicts[i][problem][keys[i]] = [igd]

    return igd_data, igd_data_dicts


def create_score_lists(data:dict, lists:bool=False) -> dict:
    '''
    Creates a dictionary where each value is the list of indicator values for the given config parameters.
    '''
    igd_plus_scores = {}
    for k,v in data.items():
        for k2,v2 in v.items():
            try:
                if lists == True:
                    igd_plus_scores[k2] += v2   
                else:
                    igd_plus_scores[k2].append(v2) 
            except:
                if lists == True:
                    igd_plus_scores[k2] = v2
                else:
                    igd_plus_scores[k2] = [v2]

    return igd_plus_scores


def create_rank_lists(data:dict) -> dict:
    '''
    rank the configurations in each problem based on indicator value
    '''
    
    igd_plus_ranks = {}
    for k,v in data.items():
        sorted_x = sorted(v.items(), key=lambda kv: kv[1])
        for i in range(len(sorted_x)):
            config = sorted_x[i][0]
            rank = i+1
            try:
                igd_plus_ranks[config].append(rank)
            except:
                igd_plus_ranks[config] = [rank]

    return igd_plus_ranks


def calculate_ranks(data:dict) -> tuple[dict, dict, dict]:
    '''
    Calculates average, median and 90th percentile IGD ranks for each key/value pair the given dictionary.
    '''

    igd_plus_ranks = create_rank_lists(data)

    # calculate the average igd+ rank for each configuration
    igd_plus_avg_ranks = {}
    igd_plus_median_ranks = {}
    igd_plus_90th_qr= {}
    for k,v in igd_plus_ranks.items():
        igd_plus_avg_ranks[k] = statistics.fmean(v)
        igd_plus_median_ranks[k] = statistics.median(v)
        igd_plus_90th_qr[k] = np.quantile(v, 0.9)

    return [igd_plus_avg_ranks, igd_plus_median_ranks, igd_plus_90th_qr]


# lists=True when the values of the dictionaries inside the dictionary contain values that are lists 
def calculate_scores(data:dict, lists:bool=False) -> tuple[dict, dict, dict]:
    '''
    Calculates average, median and 90th percentile IGD values for each key/value pair the given dictionary.
    '''
    
    igd_plus_scores = create_score_lists(data, lists=lists)

    # calculate the average igd+ values for each configuration
    igd_plus_avg_scores = {}
    igd_plus_median_scores = {}
    igd_plus_90th_qs = {}
    for k,v in igd_plus_scores.items():
        igd_plus_avg_scores[k] = statistics.fmean(v)
        igd_plus_median_scores[k] = statistics.median(v)
        igd_plus_90th_qs[k] = np.quantile(v,0.9)

    return [igd_plus_avg_scores, igd_plus_median_scores, igd_plus_90th_qs]


def do():
    '''
    Main function for running and saving all IGD analysis calculations.
    '''

    res_dict_names = [["Average regular IGD ranks", "Median regular IGD ranks", "90th percentile regular IGD rank", 
                      "Average regular IGD values", "Median regular IGD values", "90th percentile regular IGD value"],
                      ["Average IGD+ ranks", "Median IGD+ ranks", "90th percentile IGD+ rank", 
                      "Average IGD+ values", "Median IGD+ values", "90th percentile IGD+ value"]]
    
    additional_texts = [" by algorithm", " by crossover", " by mutation"]

    for k in range(2):
        # In the following line, index 1 is for IGD and 2 is for IGD+
        igd_data, igd_data_by_dict = load_data("indicator_data\\igd_values_log.txt", k+1)

        rank_data = calculate_ranks(igd_data)
        score_data = calculate_scores(igd_data)

        res_dict = rank_data + score_data

        util.save_and_print_results(res_dict, res_dict_names[k], 'igd_analysis_test\\')

        for i in range(len(igd_data_by_dict)):
            igd_data_by_ = igd_data_by_dict[i]

            rank_data_by_ = calculate_ranks(igd_data_by_)
            score_data_by_ = calculate_scores(igd_data_by_, lists=True)

            res_dict_by_ = rank_data_by_ + score_data_by_

            res_dict_names_by_ = []
            for j in range(len(res_dict_names[k])):
                res_dict_names_by_.append(res_dict_names[k][j] + additional_texts[i])

            util.save_and_print_results(res_dict_by_, res_dict_names_by_, 'igd_analysis_test\\')    


if __name__ == "__main__":
    do()
    