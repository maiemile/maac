# Code by @maiemile

from desdeo.tools.indicators_unary import hv, igd_plus_indicator, distance_indicators
from pathlib import Path
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
import utils as util
from generate_database import query_data, insert_data
import os

BASE_PATH = util.load_param_config('base_path')


def calc_ind_val_problem(run_id:int, problem_id:int, indicators:list[str], ind_vals:list[float]) -> None:
    '''
    Calculates the indicator values of the given run. Only performs calculations if the indicator value is None.
    Requires the PF approximation to exist for the given problem and the archive of the given run.
    Updates the calculated indicator values to the database.
    '''

    try:
        path = Path(BASE_PATH + 'approx_pfs/' + str(problem_id) + '.csv')
        pf_approx = np.array(pd.read_csv(path))
    except:
        # PF approximation doesn't exist, skip
        return

    # ideal and nadir vectors
    ideal_vector = np.min(pf_approx, axis=0)
    nadir_vector = np.max(pf_approx, axis=0)

    # normalize the PF approximation
    normalized_pf_approx = (pf_approx-ideal_vector) / (nadir_vector-ideal_vector)

    # it was already checked that this file exists, but for improved integrity, we use try catch
    try:
        # fetch the archive if it exists
        path = Path(BASE_PATH + 'archived_pops/' + str(run_id) + '.csv')
        archive = np.array(pd.read_csv(path))
    except:
        return

    # archive normalization
    normalized_archive = (archive-ideal_vector) / (nadir_vector-ideal_vector)

    # save the indicator values to a dictionary
    # only calculate the values if they have not been calculated already
    ind_res = {}
    for i in range(len(indicators)):
        indicator, ind_value = indicators[i], ind_vals[i]
        if ind_value != None:
            continue
        if indicator == "igd":
            ind_val = distance_indicators(normalized_archive, normalized_pf_approx).igd
        if indicator == "igd_plus":
            ind_val = igd_plus_indicator(normalized_archive, normalized_pf_approx).igd_plus

        ind_res[indicator] = ind_val

    # create the update statement
    sql = f'''UPDATE runs SET '''
    values = []
    for k,v in ind_res.items():
        sql += f'''{k} = ?,'''
        values.append(v)
    sql = sql[:-1] + f'''\nWHERE run_id = ?;'''

    # add run id to the SQL
    values.append(run_id)
    insert_data(sql, [values])


def do(indicators:list[str]) -> None:
    '''
    Function for setting up the indicator calculations with multiprocessing.
    Finds all runs and identified which ones are missing indicator values according to the given indicator list.
    Can only calculate indicator values for runs which have a saved archive and a corresponding PF approximation.
    '''

    sql_query = '''SELECT run_id, problem_id'''

    # add the indicator names to the query
    for indicator in indicators:
        sql_query += ', ' + indicator

    # get the run and problem ids, indicator values
    sql_query += ''' FROM runs'''
    data = query_data(sql_query)

    # find completed runs with missing indicator values
    # by identifying if archives have been saved for them
    completed_runs = []
    for row in data:
        if os.path.isfile(Path(BASE_PATH + 'archived_pops/' + str(row[0]) + '.csv')):
            new_row = list(row)

            # if there is at least one missing indicator value, add the row to the list
            if None in new_row[2:]:
                completed_runs.append(new_row[:2] + [indicators] + [new_row[2:]])

    #print(completed_runs)

    with Pool(processes=cpu_count()) as pool:
        pool.starmap(calc_ind_val_problem, completed_runs)
        pool.terminate()
        pool.join()
