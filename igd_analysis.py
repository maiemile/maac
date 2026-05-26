# Code by @maiemile

import statistics
import numpy as np
import utils as util

# igd_data is a dictionary of dictionaries: 
# each problem is a key in the dictionary
# the value of which is a dictionary of
# configuration: igd/igd+ values

def load_data(file_name, index_of_data):
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

def create_score_lists(data, lists=False):
    # igd_plus_scores:
    # key = configuration,
    # value = list of igd+ values of that configuration
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

def create_rank_lists(data):
    # rank the configurations in each problem based on IGD+ value
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

def calculate_ranks(data):

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
def calculate_scores(data, lists=False):
    
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
    

if __name__ == "__main__":

    # IMPORTANT! 
    # In the following line, Use 1 for IGD and 2 for IGD+
    igd_data, igd_data_by_dict = load_data("indicator_data\\igd_values_log.txt", 1)

    #res_dict_names = ["Average IGD+ ranks", "Median IGD+ ranks", "90th percentile IGD+ rank", 
    #                  "Average IGD+ values", "Median IGD+ values", "90th percentile IGD+ value"]

    res_dict_names = ["Average regular IGD ranks", "Median regular IGD ranks", "90th percentile regular IGD rank", 
                      "Average regular IGD values", "Median regular IGD values", "90th percentile regular IGD value"]
    
    rank_data = calculate_ranks(igd_data)
    score_data = calculate_scores(igd_data)
    
    res_dict = rank_data + score_data
    
    util.save_and_print_results(res_dict, res_dict_names, 'igd_analysis\\')
    
    additional_texts = [" by algorithm", " by crossover", " by mutation"]
    for i in range(len(igd_data_by_dict)):
        igd_data_by_ = igd_data_by_dict[i]
    
        rank_data_by_ = calculate_ranks(igd_data_by_)
        score_data_by_ = calculate_scores(igd_data_by_, lists=True)
    
        res_dict_by_ = rank_data_by_ + score_data_by_
    
        res_dict_names_by_ = []
        for j in range(len(res_dict_names)):
            res_dict_names_by_.append(res_dict_names[j] + additional_texts[i])
    
        util.save_and_print_results(res_dict_by_, res_dict_names_by_, 'igd_analysis\\')