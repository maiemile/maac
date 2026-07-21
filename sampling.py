# Code by @maiemile

from scipy.stats import qmc, entropy
from pymoo.problems import get_problem
import numpy as np
from pflacco.classical_ela_features import (calculate_ela_meta, calculate_ela_distribution, calculate_pca, calculate_nbc,
                                            calculate_dispersion, calculate_information_content, calculate_ela_level)
from desdeo.tools.non_dominated_sorting import fast_non_dominated_sort_indices
from scipy.spatial.distance import pdist
import utils as util
from generate_database import query_data, generate_feature_table, insert_data

re_problems = util.get_re_problems()


def sample_problem(problem:tuple[int,str,int,int]):
    '''
    Creates a sample of the given multi-objective optimization problem.
    Uses latin hypercube sampling to sample in the decision space
    and the decision vectors are evaluated to get the objective vectors.
    Works for RE, WFG and DTLZ problem suites.
    '''
    prob_name = problem[1]
    n_obj = problem[2]
    n_var = problem[3]

    # get the lower and upper bounds of the problem
    if prob_name[:2] == 're':
        problem_class = re_problems[prob_name]
        lowerbound = problem_class().lbound
        upperbound = problem_class().ubound
    if prob_name[:3] == 'wfg':
        lowerbound = [0.0] * n_var
        upperbound = [1.0*2*(i+1) for i in range(n_var)]
    if prob_name[:4] == 'dtlz':
        lowerbound = [0.0] * n_var
        upperbound = [1.0] * n_var


    # generate a latin hypercube sample with 200*n_var samples
    rng = np.random.default_rng(seed=42)
    sampler = qmc.LatinHypercube(d=n_var, rng=rng)
    n = 200*n_var
    sample = sampler.random(n=n) # This number is from the article "Landscape Features and Automated Algorithm Selection for Multi-objective Interpolated Continuous Optimisation Problems"
    fixed_sample = qmc.scale(sample, lowerbound, upperbound)

    # evaluate the samples
    if prob_name[:2] == 're':
        evaluated = []
        prob = problem_class()
        for row in fixed_sample:
            evaluated.append(prob.evaluate(row).tolist())
        evaluated = np.array(evaluated)
    # TODO: use the built-in DESDEO constructor for pymoo problems, evaluate directly, can fetch lower/upper bounds from the object earlier?
    else:
        problem_pymoo = get_problem(prob_name, n_var=n_var, n_obj=n_obj)
        out = {}
        problem_pymoo._evaluate(fixed_sample, out)
        evaluated = out["F"]
    return fixed_sample, evaluated


def calculate_ela_features(X:np.array, y:np.array) -> dict:
    '''
    Calculates 7 clasical single-objective exploratory landscape analysis feature sets from pflacco.
    '''
    # calculate 7 feature sets and combine the results to a dictionary
    ela_meta = calculate_ela_meta(X,y)
    ela_dict = ela_meta
    pca = calculate_pca(X,y)
    ela_dict = ela_dict | pca
    nbc = calculate_nbc(X,y)
    ela_dict = ela_dict | nbc
    disp = calculate_dispersion(X,y)
    ela_dict = ela_dict | disp
    ic = calculate_information_content(X,y)
    ela_dict = ela_dict | ic
    ela_dist = calculate_ela_distribution(X,y)
    ela_dict = ela_dict | ela_dist
    ela_level = calculate_ela_level(X,y)
    ela_dict = ela_dict | ela_level

    return ela_dict


def calculate_moo_features(X:np.array, y:np.array, nds_indices:list[list[int]]) -> dict:
    '''
    Calculates some multi-objective specific exploratory landscape analysis features proposed by the following paper:

    Arnaud Liefooghe, Sébastien Verel, Benjamin Lacroix, Alexandru-Ciprian Zăvoianu, and John McCall. 2021. 
    Landscape features and automated algorithm selection for multi-objective interpolated continuous optimisation problems. 
    In Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '21). Association for Computing Machinery, 
    New York, NY, USA, 421–429. https://doi.org/10.1145/3449639.3459353
    '''

    # calculate how many solutions there are per non-dominated front
    samples_per_front = []
    sum_of_ranks = 0
    for i in range(len(nds_indices)):
        samples_per_front.append(len(nds_indices[i]))
        sum_of_ranks += (i+1)*len(nds_indices[i])

    # calculate the proportion of solutions in each front
    prob_per_front = [value/sum(samples_per_front) for value in samples_per_front]

    mo_ela_dict = {}

    proportion_of_non_dominated_solutions = prob_per_front[0] # FEAT: nd_n
    mo_ela_dict['nd_n'] = proportion_of_non_dominated_solutions

    maximum_front_number = len(nds_indices) # FEAT: rank_max
    mo_ela_dict['rank_max'] = maximum_front_number

    average_front_number = sum_of_ranks / sum(samples_per_front) # FEAT: rank_avg
    mo_ela_dict['rank_avg'] = average_front_number

    entropy_nds = entropy(prob_per_front) # FEAT: rank_ent
    mo_ela_dict['rank_ent'] = entropy_nds

    # average and maximum distances between decision vectors
    dists_var = pdist(X)
    dist_x_avg = np.mean(dists_var)
    mo_ela_dict['dist_x_avg'] = dist_x_avg
    dist_x_max = np.max(dists_var)
    mo_ela_dict['dist_x_max'] = dist_x_max

    # average and maximum distances between objective vectors
    dists_obj = pdist(y)
    dist_f_avg = np.mean(dists_obj)
    mo_ela_dict['dist_f_avg'] = dist_f_avg
    dist_f_max = np.max(dists_obj)
    mo_ela_dict['dist_f_max'] = dist_f_max

    # average and maximum distances between non-dominated decision vectors
    non_dominated_solutions = X[nds_indices[0]]
    dists_nd = pdist(non_dominated_solutions)
    dist_x_nd_avg = np.mean(dists_nd)
    mo_ela_dict['dist_x_nd_avg'] = dist_x_nd_avg
    dist_x_nd_max = np.max(dists_nd)
    mo_ela_dict['dist_x_nd_max'] = dist_x_nd_max

    return mo_ela_dict


def do(aggregators:list[str] = None):
    '''
    The default function for running the sampling procedures.

    :param aggregators: List of aggregation strategies to use. Available strategies/high-level feature sets: 
        'max', 'min', 'avg', 'sd', 'nds', 'moo' 
    '''

    # load all problem instances from the database
    sql_query = '''SELECT * from problems'''
    problem_instances = query_data(sql_query)

    # if no aggregators were given, use the defaults
    if aggregators == None:
        aggregators = util.get_default_aggregators()

    table_created = False
    sql = None

    # loop through all problems and calculate the ELA features
    for prob in problem_instances:
        X, y = sample_problem(prob)
        
        dictionaries = []
        # calculate the features one objective function at a time
        # Currently, 7 feature sets can be calculated without 
        # explicitly giving the function or errors
        for i in range(len(y[0])):
            ela_dict = calculate_ela_features(X,y[:,i])

            dictionaries.append(ela_dict)

        max_dict = {}
        min_dict = {}
        avg_dict = {}
        sd_dict = {}

        # for each feature, calculate the max, min, avg and standard deviation
        for key in dictionaries[0].keys():
            max_dict[key] = max(d[key] for d in dictionaries)
            min_dict[key] = min(d[key] for d in dictionaries)
            avg_dict[key] = sum(d[key] for d in dictionaries) / len(dictionaries)
            values = np.array([d[key] for d in dictionaries])
            sd_dict[key] = np.std(values)

        # do non-dominated sorting on the sample 
        # then, use the front numbers as the "scalarized" objective functions to calculate ELA features
        nds_indices = fast_non_dominated_sort_indices(y)
        front_numbers = np.empty(len(y))

        # fill the above empty array with the front numbers from th NDS data
        for i in range(len(nds_indices)):
            for j in range(len(nds_indices[i])):
                index = nds_indices[i][j]
                front_numbers[index] = i+1

        #calculate the features on the NDS data
        nds_ela_dict = calculate_ela_features(X,front_numbers)

        #calculate features specific to multi-objective optimization
        moo_ela_dict = calculate_moo_features(X,y,nds_indices)

        dict_names = {"max":max_dict, "min": min_dict, "avg": avg_dict, "sd": sd_dict, "nds": nds_ela_dict, "moo": moo_ela_dict}
        dicts = [dict_names[agg] for agg in aggregators]

        # create lists of feature names and values
        feature_names = []
        feature_values = []
        for i in range(len(dicts)):
            for k, v in dicts[i].items():
                # runtime isn't a feature
                if 'runtime' in k:
                    continue
                feature_name = k + '_' + aggregators[i]
                # replace dots with underscores for SQLite
                feature_names.append(feature_name.replace('.', '_'))
                feature_values.append(v)
        
        # check that the features table exists and fetch the SQL for data insertion
        if not table_created:
            sql = generate_feature_table(feature_names)
            table_created = True

        prob_id = prob[0]
        # insert a row of data
        insert_data(sql, [[int(prob_id)] + feature_values])


if __name__ == "__main__":
    do()    

            
