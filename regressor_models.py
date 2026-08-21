# code by @maiemile

import pandas as pd
import numpy as np
import sqlite3
import xgboost as xgb
import pickle
import os
from pathlib import Path

from generate_database import get_best_config_by_median, query_data
import utils as util

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import (r2_score, mean_squared_error, mean_squared_log_error, mean_absolute_percentage_error)

# Fetch the information on whether to load pre-existing models (True) or train new ones (False)
load_models = bool(util.load_param_config('load_models'))
regr_name = str(util.load_param_config('regressor_name'))
# Load the filename of the database and the base path
database = util.load_param_config('database_file')


def get_model_data() -> dict:
    '''
    Returns the default machine learning models and their parameter options for hyperparameter optimization.
    '''
    # models and their parameter grid for grid search with cross-validation
    regr_rf = RandomForestRegressor(random_state=0)
    regr_dt = DecisionTreeRegressor(random_state=0) 
    regr_xg = xgb.XGBRegressor(random_state=0)
    regr_nn = MLPRegressor(random_state=0, max_iter=500)
    param_grid_rf = {
        "n_estimators": [10,50,100,200],
        "criterion": ["squared_error", "poisson"],
        "max_depth": [None, 2,4,7],
        "max_features": [None, "sqrt", "log2"],
    }
    param_grid_dt = {
        "criterion": ["squared_error", "absolute_error", "poisson"],
        "max_depth": [None, 3,5,10],
        "max_features": [None, "sqrt", "log2"],
        "splitter": ["best", "random"]
    }
    param_grid_xg = {
        "max_depth": [6,8,10,12],
        "subsample": [0.5, 0.75, 1],
        "eta": [0.01, 0.1, 0.3, 0.6],
        "n_estimators": [10,50,100,200],
    }
    param_grid_nn = {
        "hidden_layer_sizes": [(30,10,6), (20,12,4), (50, 30, 10, 4), (16,6), (12,4)],
        "solver": ["adam", "lbfgs"],
        "learning_rate": ["constant", "adaptive"],
        "activation": ["logistic", "relu"]
    }

    model_dict = {
        "Random forest": [regr_rf, param_grid_rf], 
        "Decision tree": [regr_dt, param_grid_dt], 
        "XGBoost": [regr_xg, param_grid_xg], # TODO: loading XGBoost doesn't work sometimes
        "Neural network": [regr_nn, param_grid_nn]
    }
    
    return model_dict


def calculate_r2_scores(data, indicator:str):
    '''
    Calculates the R2 scores for regression models predicting the IGD value.
    Saves the results in a .txt file.
    '''

    # get all EA ids
    sql = '''SELECT ea_id FROM eas'''
    res = query_data(sql)

    X_train, X_test, y_train, y_test = data

    # concat the data
    X = pd.concat([X_train, X_test])
    y = pd.concat([y_train, y_test])

    # reset index to prevent any issues with indices
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    dataf = pd.concat([X,y], axis=1)

    # loop through all configurations to get their R2 scores
    r2_scores = []
    for config in res:
        ea = config[0]

        # Load only rows that contain the correct configuration
        index = "ea_id_" + str(ea)
        dataf_prob = dataf[dataf[index] == 1.0]

        # Run a default random forest regressor on the data
        regr = RandomForestRegressor(random_state=0)

        y_temp = dataf_prob[indicator]
        X_temp = dataf_prob.drop([indicator], axis=1)

        # Fit the model and calculate R2 scores on the train data
        regr.fit(X_temp, y_temp)
        y_pred = regr.predict(X_temp)
        r2_pred = r2_score(y_temp, y_pred)
        r2_scores.append((ea, r2_pred))

    # Sort the configs based on the R2 scores in descending order
    sorted_r2_scores = sorted(r2_scores, key=lambda x: x[1], reverse=True)  

    for item in sorted_r2_scores:
        print(item[0], item[1])

    # save the sorted results to text files
    path = Path('model_analysis/r2_scores_regr.txt')
    with open(path, "w") as file:
        for line in sorted_r2_scores:
            file.write(" ".join(str(item) for item in line) + "\n")

    return sorted_r2_scores


def select_features(X_temp, y) -> list[int]:
    '''
    Select the most important features and return the indexes of these features.
    '''

    # TODO: k should be a config parameter
    selector = SelectKBest(f_regression, k=50)
    _ = selector.fit_transform(X_temp, y)
    cols_idxs = list(selector.get_support(indices=True))

    return cols_idxs


def feature_selection(df_to_normalize, X_train, y_train, enc) -> tuple[list[int], list[str]]:
    '''
    Perform feature selection for regressor models.
    '''
    # if we want to load pre-existing features for the model
    if load_models:
        # Try to find a regressor model in the models folder
        modelname = None
        for x in os.listdir('models'):
            # TODO: could allow other file types, model chosen by user?
            if x.endswith(".pkl"):
                if '_regressor' in x:
                    modelname = x
                    break
        
        # If no classifier models were found, raise an exception
        if modelname == None:
            raise Exception('No models found. Feature names could not be loaded.')
        
        with open(Path(f'models/{modelname}'), 'rb') as f:
            best_estimator = pickle.load(f)
            feature_names = best_estimator.feature_names_in_
            # get the indexes of the existing features in the dataframe
            selected_columns = df_to_normalize.columns.get_indexer(feature_names)
            # remove the extra columns not found in the dataframe that contains the numerical columns
            selected_columns = [x for x in selected_columns if x != -1]
    else:
        # select the most important features using only the landscape features, because we want to keep the config as an input feature
        selected_columns = select_features(X_train, y_train)
        encoded_feature_names = list(enc.get_feature_names_out())
        selected_column_names = list(df_to_normalize.columns[selected_columns])
        feature_names = selected_column_names + encoded_feature_names

    return selected_columns, feature_names


def prepare_data(df: pd.DataFrame, test_problems: list[int], scaler, 
                 enc, response_variable: str) -> tuple[list, pd.DataFrame]:
    '''
    Does data preprocessing. Missing and unrealistic values are handled. 
    Categorical variables are encoded, numerical variables are scaled and train/test split is performed.
    Features are selected based on importance or loaded from existing models.
    '''

    # Remove all columns with missing or infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=1)

    # train/test split
    df_test = df[df['problem_id'].isin(test_problems)]
    df = df.drop(df[df['problem_id'].isin(test_problems)].index)

    # remove unnecessary features
    problem_data = ['problem_id', 'seed']
    df = df.drop(problem_data, axis=1)
    # save the data related to the run in a separate dataframe
    df_test_data = df_test[problem_data + ['ea_id']]
    # reset the index to be in line with the X_test dataframe
    df_test_data.reset_index(drop=True, inplace=True)
    df_test = df_test.drop(problem_data, axis=1)

    y_cols = [response_variable]
    cat_vars = ['ea_id']

    # scale or encode train data depending on if the feature is numerical or categorical
    categorical_vars = df[cat_vars]
    X2 = enc.fit_transform(categorical_vars).toarray()
    df_to_normalize = df.drop(columns=y_cols+cat_vars)
    # fix binary columns TODO: change the format of these columns in the database / store them as floats in ELA calculations
    df_to_normalize = util.convert_data(df_to_normalize)
    X1 = scaler.fit_transform(df_to_normalize)
    y_train = df[y_cols]

    # scale or encode test data depending on if the feature is numerical or categorical
    categorical_vars_test = df_test[cat_vars]
    X2_test = enc.transform(categorical_vars_test).toarray()
    df_to_normalize = df_test.drop(columns=y_cols+cat_vars)
    df_to_normalize = util.convert_data(df_to_normalize)
    X1_test = scaler.transform(df_to_normalize)
    y_test = df_test[y_cols]

    selected_columns, feature_names = feature_selection(df_to_normalize, X1, y_train, enc)

    # update the datasets to only include the most important features
    X1_new = X1[:,selected_columns]
    X_train = np.hstack((X1_new, X2))
    X_train = pd.DataFrame(X_train, columns=feature_names)

    X1_test_new = X1_test[:,selected_columns]
    X_test = np.hstack((X1_test_new, X2_test))
    X_test = pd.DataFrame(X_test, columns=feature_names)

    return [X_train, X_test, y_train, y_test], df_test_data


def optimize_models(regr, X_train, y_train, param_grid:dict):
    '''
    Performs hyperparameter optimization for the given regression model and its parameter grid
    '''

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=regr,
        param_grid=param_grid,
        cv=kfold,
        scoring='neg_mean_absolute_percentage_error', # negative MAPE because grid search tries to maximize the score
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    print("Best parameters:", grid_search.best_params_)
    print("Best CV score (MAPE):", -grid_search.best_score_)

    best_estimator = grid_search.best_estimator_

    return best_estimator


def run_regression_models(test_problems: list[str], model, data:list,
                              param_grid:dict, model_name: str, response_variable: str, 
                              prob_data:pd.DataFrame, single_best_solver:int, 
                              load_file:bool=False) -> tuple[float, list[float], list[float], list[float]]:
    '''
    Train and test the regression model, including hyperparameter optimization using cross-validation.
    Creates confusion matrices of the results.
    '''
    X_train, X_test, y_train, y_test = data
    regr = model

    # get the parameter names
    sql ='''SELECT name FROM PRAGMA_TABLE_INFO('eas') WHERE name!='ea_id' AND name!='name' '''
    res = query_data(sql)

    # reformulate the parameter names for SQL
    params = ''
    params_list = []
    for item in res:
        params = params+item[0]+','
        params_list.append(item[0])
    params = params[:-1]
    #print(params)

    if load_file:
        # Load the model
        with open(Path(f'models/{model_name}_regressor{regr_name}.pkl'), 'rb') as f:
            best_estimator = pickle.load(f)
    else:
        # do hyperparameter optimization for the chosen regressor model
        best_estimator = optimize_models(regr, X_train, y_train, param_grid)
    
        # save the model
        with open(Path(f'models/{model_name}_regressor{regr_name}.pkl'),'wb') as f:
            pickle.dump(best_estimator,f)

    # Calculate MSE on the test data
    y_pred = best_estimator.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mpe = mean_absolute_percentage_error(y_test, y_pred)

    print(mse, mpe)

    # TODO: could be a dictionary?
    igd_values = []
    sbs_igd_values = []
    vbs_igd_values = []

    # fetch the optimal configuration for each problem
    optimal_configs = get_best_config_by_median()
    
    optimal_configs_test = []
    predicted_configs_test = []

    # create a dataframe by combining the predicted values and the run information
    y_pred_s = pd.Series(y_pred, name=response_variable)
    new_df = pd.concat([prob_data, y_pred_s], axis=1)
    
    # loop through the test problems to analyze
    for problem in test_problems:
        seed = 1
        while True:
            problem_data = new_df.loc[(new_df['problem_id'] == problem) & (new_df['seed'] == seed)]
            if len(problem_data) < 1:
                break
            # find the configuration with the best IGD value
            best_ea = int(problem_data.iloc[np.argmin(problem_data[response_variable])]["ea_id"])

            # load the predicted configuration parameters
            sql = f'''SELECT {params} FROM eas WHERE ea_id = {best_ea}'''
            res = query_data(sql)

            # and the best configuration parameters (virtual best solver)
            true_best = optimal_configs[problem][0]
            true_best_igd = optimal_configs[problem][1]
            vbs_igd_values.append(true_best_igd)
            sql = f'''SELECT {params} FROM eas WHERE ea_id = {true_best}'''
            res_true = query_data(sql)

            #print('-----------------------------')
            #print(f"Prediction for problem {problem} with seed {seed}: EA with ID: {best_ea}, components {res}")
            #print("True best", res_true)
            #print('-----------------------------')

            optimal_configs_test.append(res_true)
            predicted_configs_test.append(res)

            # get the IGD value of the predicted configuration
            sql = f'''SELECT {response_variable} FROM runs WHERE ea_id = {best_ea} AND problem_id = {problem}'''
            res = query_data(sql)
            # TODO: temporary solution to handle None values
            # RVEA-NUM must be handled/ignored properly...
            for i in range(len(res)):
                if res[i][0] == None:
                    res[i] = (9999999,)
 
            median_igd = np.median(np.array(res))
            igd_values.append(median_igd)

            # get the IGD value of the SBS
            sql = f'''SELECT {response_variable} FROM runs WHERE ea_id = {single_best_solver} AND problem_id = {problem}'''
            res = query_data(sql)
            median_igd = np.median(np.array(res))
            sbs_igd_values.append(median_igd)
            seed += 1

    for igd, sbs in zip(igd_values, sbs_igd_values):
        print(f"IGD of the predicted configuration: {igd}, IGD of the SBS: {sbs}")

    # create and save confusion matrices of the predicted parameters of the configurations
    util.create_confusion_matrices(np.asarray(optimal_configs_test), np.asarray(predicted_configs_test), model_name+'_regressor', params_list)

    return mpe, igd_values, sbs_igd_values, vbs_igd_values


def do(model_dict: dict = None, configs: list[str] = None, indicator: str = None) -> None:
    '''
    Default function for running the full pipeline of running the regression-based configurator models.
    '''

    if model_dict == None:
        model_dict = get_model_data()

    if indicator == None:
        # Load the default indicator
        indicator = util.load_param_config('indicator')

    # fetch the ids of problems used in the testing phase
    test_prob = ["dtlz2", "wfg7", "re31", "re32", "re33", "re34", "re37", "re41", "re42", "re61"] 
    test_problems = util.get_test_problems(test_prob)

    enc = OneHotEncoder(handle_unknown='ignore')
    scaler = StandardScaler()

    # load the single best solver, ignoring the test problems
    single_best_solver = util.determine_single_best_solver(test_problems=test_problems)

    # Combine data from runs and features tables into one pandas dataframe
    try:  
        con = sqlite3.connect(database)
        sql = f"""SELECT f.*, r.ea_id, r.{indicator} FROM features f 
                    JOIN runs r ON r.problem_id = f.problem_id AND r.seed = f.seed WHERE r.{indicator} IS NOT NULL"""
        df = pd.read_sql_query(sql, con)
    except sqlite3.Error as e:
        print(e)
    finally:
        con.close()

    data, problem_data = prepare_data(df, test_problems, scaler, enc, indicator)

    #r2_scores = calculate_r2_scores(data, indicator)

    igd_value_sets = []
    config_labels = []
    mse_values = {}

    # loop through the models and run all of them on the data, create performance profile plots of the results
    for model_name, model_data in model_dict.items():
        model, param_grid = model_data
        print("-"*10, model_name, "-"*10)
        
        # run all of the regression models, either with hyperparameter optimization or using existing models
        mse, igd_values, sbs_igd_values, vbs_igd_values = run_regression_models(test_problems, model, data, param_grid, model_name,
                                                     indicator, problem_data, single_best_solver, load_file=load_models)
        config_results = [vbs_igd_values, sbs_igd_values, igd_values]
        # convert to a dataframe
        configs = ['VBS', 'SBS', model_name + ' regressor']
        igd_df = pd.DataFrame(np.array(config_results).T, columns=configs)

        util.create_performance_profile_plot(igd_df, configs, model_name + '_regressor') 
        igd_value_sets.append(igd_values)
        config_labels.append(model_name + ' regressor')
        mse_values[model_name] = mse

    # performance profile plot for comparing the configurators against each other
    igd_df = pd.DataFrame(np.array(igd_value_sets).T, columns=config_labels)
    util.create_performance_profile_plot(igd_df, config_labels, 'regressors', font_size=6)

    for model, mse_value in mse_values.items():
        print(model, mse_value)


if __name__ == "__main__":
    do()