# Code by @maiemile

import numpy as np
import pandas as pd
import utils as util
import xgboost as xgb
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import scienceplots

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (r2_score, mean_squared_error, mean_squared_log_error, mean_absolute_percentage_error
                             , confusion_matrix ,ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, GridSearchCV
plt.style.use('science')

# all the problems instances
problem_instances = util.get_problem_instances()
algos, crossovers, mutations = util.get_all_configuration_options()

cols_to_drop = ['problem', 'igd_plus', 'ic.eps_ratio_MIN', 'ic.eps_ratio_AVG', 'ic.eps_ratio_SD']

# Fetch the information on whether to load pre-existing models (True) or train new ones (False)
load_models = util.load_files_config()


def calculate_r2_scores(feat_sets, response_variable:str):
    '''
    Calculates the R2 scores for regression models predicting the IGD value.
    Saves the results in a .txt file.
    '''
    igd_labels = ['problem', 'config', 'igd', 'igd_plus', 'objectives', 'variables']

    # get the necessary data for running the models
    igd_array = create_igd_array('indicator_data\\igd_values_log.txt')
    labels = util.get_labels_from_file(igd_labels, feat_sets)
    dataframe = get_dataframe(igd_array, labels, feat_sets, problem_instances)

    #get the R2 scores of the models 
    r2_scores = run_models(dataframe, response_variable)

    # Sort the configs based on the R2 scores in descending order
    sorted_r2_scores = sorted(r2_scores, key=lambda x: x[1], reverse=True)  

    for item in sorted_r2_scores:
        print(item[0], item[1])

    # save the sorted results to text files
    path = Path(f'model_analysis\\r2_scores_regressor2.txt')
    with open(path, "w") as file:
        for line in sorted_r2_scores:
            file.write(" ".join(str(item) for item in line) + "\n")


def create_igd_array(file_name):
    igd_array, _, _ = util.create_igd_array_and_dict(file_name)
    return igd_array


def get_dataframe(igd_array, igd_labels, feat_sets, problem_instances):
    igd_data = util.load_data(igd_array, feat_sets, problem_instances)
    df = pd.DataFrame(igd_data, columns=igd_labels)
    return df


def select_features(X_temp, y):
    # Select 100 most important features for each response variable separately
    # and concatenate the chosen features
    selector = SelectKBest(f_regression, k=100)
    _ = selector.fit_transform(X_temp, y)
    cols_idxs = list(selector.get_support(indices=True))

    return cols_idxs


def run_models(df, response_variable:str):
    configs = [a+'-'+c+'-'+m for a in algos for c in crossovers for m in mutations]

    r2_score_by_config = []

    y_cols_ = [response_variable]
    y_cols = y_cols_ + ['problem', 'config', 'igd_plus', 'ic.eps_ratio_MIN', 'ic.eps_ratio_AVG', 'ic.eps_ratio_SD']
    df_config_temp = df.drop(columns=y_cols)
    scaler = StandardScaler().fit(df_config_temp)

    y_cols_2 = ['problem', 'igd_plus', 'ic.eps_ratio_MIN', 'ic.eps_ratio_AVG', 'ic.eps_ratio_SD']
    df_config = df.drop(columns=y_cols_2)

    for config in configs:
        # Load only rows that contain the correct configuration
        df_config_c = df_config.loc[df_config['config'] == config]
        # Separate igd column into response variable
        y = np.array(df_config_c[y_cols_]).ravel()

        # drop unnecessary columns from the dataframe to create the input variables
        df_config_c = df_config_c.drop(columns=y_cols_ + ['config'])
        column_names = list(df_config_c)
        X = scaler.transform(df_config_c)
        X = pd.DataFrame(X, columns=column_names)

        # Run a default random forest regressor on the data
        regr = RandomForestRegressor(random_state=0)

        # Fit the model and calculate R2 scores on the train data
        regr.fit(X,y)
        y_pred = regr.predict(X)
        r2_pred = r2_score(y, y_pred)
        r2_score_by_config.append((config, r2_pred))

    return r2_score_by_config


def prepare_data(df, test_problems, scaler, enc, response_variable: str, load_features: bool=False):

    # get the test set from the full data based on the names of the test problems
    test_set = df[df['problem'].str.contains('|'.join(test_problems), na=False)]

    df = df.drop(df[df['problem'].isin(test_problems)].index)

    y_cols = [response_variable]
    cat_vars = ['algorithm', 'crossover', 'mutation']

    # scale or encode train data depending on if the feature is numerical or categorical
    categorical_vars = df[cat_vars]
    X2 = enc.fit_transform(categorical_vars).toarray()
    df_to_normalize = df.drop(columns=y_cols+cat_vars+cols_to_drop)
    X1 = scaler.fit_transform(df_to_normalize)
    y_train = df[y_cols]

    # scale or encode test data depending on if the feature is numerical or categorical
    categorical_vars_test = test_set[cat_vars]
    X2_test = enc.transform(categorical_vars_test).toarray()
    df_to_normalize = test_set.drop(columns=y_cols+cat_vars+cols_to_drop)
    X1_test = scaler.transform(df_to_normalize)
    y_test = test_set[y_cols]

    # if we want to load pre-existing features for the model
    # TODO: this should be improved to work if there is another model available/use model defined by user
    if load_features:
        with open(f'models\\Random forest_regressor.pkl', 'rb') as f:
            best_estimator = pickle.load(f)
            feature_names = best_estimator.feature_names_in_
            # get the indexes of the existing features in the dataframe
            selected_columns = df_to_normalize.columns.get_indexer(feature_names)
            # remove the extra columns not found in the dataframe that contains the numerical columns
            selected_columns = [x for x in selected_columns if x != -1]
    else:
        # select the most important features using only the landscape features, because we want to keep the config as an input feature
        selected_columns = select_features(X1, y_train)
        encoded_feature_names = list(enc.get_feature_names_out())
        selected_column_names = list(df_to_normalize.columns[selected_columns])
        feature_names = selected_column_names + encoded_feature_names
    
    # update the datasets to only include the most important features
    X1_new = X1[:,selected_columns]
    X_train = np.hstack((X1_new, X2))
    X_train = pd.DataFrame(X_train, columns=feature_names)

    X1_test_new = X1_test[:,selected_columns]
    X_test = np.hstack((X1_test_new, X2_test))
    X_test = pd.DataFrame(X_test, columns=feature_names)

    return [X_train, X_test, y_train, y_test], selected_columns


def get_model_data():
    # models and their parameter grid for grid search with cross-validation
    regr_rf = RandomForestRegressor(random_state=0)
    regr_dt = DecisionTreeRegressor(random_state=0) 
    regr_xg = xgb.XGBRegressor(random_state=0)
    regr_nn = MLPRegressor(random_state=0, max_iter=500)
    param_grid_rf = {
        "n_estimators": [10,50,100,200],
        "criterion": ["squared_error", "friedman_mse", "poisson"],
        "max_depth": [None, 2,4,7],
        "max_features": [None, "sqrt", "log2"],
    }
    param_grid_dt = {
        "criterion": ["squared_error", "friedman_mse", "poisson"],
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
        "XGBoost": [regr_xg, param_grid_xg],
        "Neural network": [regr_nn, param_grid_nn]
    }
    
    return model_dict


def optimize_models(regr, X_train, y_train, param_grid):
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)

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


def get_optimal_configs(problems_to_ignore):
    Y = {}

    # load the response variables (the optimal configuration for each problem)
    with open('indicator_data\\best_regular_igd_values.txt', 'r') as file:
        for line in file:
            split_line = line.split() 
            problem = split_line[0]
            # ignore the troublesome problems
            if problem in problems_to_ignore:
                continue
            configuration = split_line[1]
            split_config = configuration.split('-')
            Y[problem] = split_config

    return Y


def create_confusion_matrices(y_test, y_pred_test, model_name):
    # Confusion matrix for each output
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,4))
    axes = [ax1, ax2, ax3]
    print(y_test, y_pred_test)
    for j in range(len(y_test[0])):
        cm = confusion_matrix(y_test[:, j], y_pred_test[:, j])
        print(cm)
        # A "dumb" way to deal with rvea not appearing in the test set or predictions
        if j != 0:
            if j == 1:
                labels = [r"BLX-$\alpha$", "LX", "SBX", "SAX"]
            if j == 2:
                labels = ["BPM", "MPTM", "NUM", "PM"]
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=labels)
        else:
            if "rvea" in y_pred_test[:, j]:
                labels = ["IBEA", "NSGA-III", "RVEA"]
                disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=labels)
            else:
                labels = ["IBEA", "NSGA-III"]
                disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=labels)
        
        disp.plot(ax=axes[j], colorbar=False)

    ax1.set_title('Algorithm')
    ax2.set_title('Crossover operator')
    ax3.set_title('Mutation operator')
    ax2.set_ylabel('')
    ax3.set_ylabel('')
    
    plt.savefig(f'figures\\confusion_matrices\\{model_name}_regr.pdf')
    plt.show()



def run_full_regression_model(df, igd_dict, test_problems, model, data, scaler, enc, selected_columns, param_grid, model_name: str, 
                              problems_to_ignore: list[str], response_variable: str, load_file=False):

    test_set = df[df['problem'].str.contains('|'.join(test_problems), na=False)]
    y_cols = [response_variable]
    cat_vars = ['algorithm', 'crossover', 'mutation']

    X_train, X_test, y_train, y_test = data

    regr = model

    if load_file:
        # Load the model
        with open(f'models\\{model_name}_regressor.pkl', 'rb') as f:
            best_estimator = pickle.load(f)
    else:
        # do hyperparameter optimization for the chosen regressor model
        best_estimator = optimize_models(regr, X_train, y_train, param_grid)
    
        # save the model
        with open(f'models\\{model_name}_regressor.pkl','wb') as f:
            pickle.dump(best_estimator,f)


    # Calculate MSE on the test data
    y_pred = best_estimator.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    #mle = mean_squared_log_error(y_test, y_pred)
    mpe = mean_absolute_percentage_error(y_test, y_pred)

    print(mse, mpe)

    problems_and_configs = []
    igd_values = []
    
    optimal_configs = get_optimal_configs(problems_to_ignore)

    optimal_configs_test = []
    predicted_configs_test = []

    # loop through the test problems to analyze
    for problem in test_problems:
        problem_data = test_set.loc[test_set['problem'] == problem]

        # transform the data
        catvarst = problem_data[cat_vars]
        X2t = enc.transform(catvarst).toarray()
        df_to_normalizet = problem_data.drop(columns=y_cols+cat_vars+cols_to_drop)
        X1t = scaler.transform(df_to_normalizet)
        X1t_new = X1t[:,selected_columns]
        Xt = np.hstack((X1t_new, X2t))
        yt = problem_data[y_cols]

        # use the model to make predictions
        y_predt = best_estimator.predict(Xt)

        # find the configuration with the best IGD value
        best_index = np.argmin(y_predt)
        algo = problem_data.iloc[best_index]["algorithm"]
        crossover = problem_data.iloc[best_index]["crossover"]
        mutation = problem_data.iloc[best_index]["mutation"]

        # save the best config
        problem_plus_config = problem + '-' + algo + '-' + crossover + '-' + mutation
        problems_and_configs.append(problem_plus_config)

        print(problem_plus_config, igd_dict[problem_plus_config][0])

        igd_values.append(igd_dict[problem_plus_config][0])

        # save the optimal configuration of the problem and the predicted configuration to lists
        optimal_configs_test.append(optimal_configs[problem])
        predicted_configs_test.append([algo, crossover, mutation])

    # create and save confusion matrices of the predicted parameters of the configurations
    create_confusion_matrices(np.asarray(optimal_configs_test), np.asarray(predicted_configs_test), model_name)

    return mse, igd_values


def do(model_dict: dict = None, feat_sets: list[str] = None, configs: list[str] = None, response_variable: str = "igd", 
       problems_to_ignore: list[str] = []) -> None:
    # fetch the names of problems used in the testing phase
    test_problems = util.get_test_problems()

    igd_labels_2 = ['problem', 'algorithm', 'crossover', 'mutation', 'igd', 'igd_plus', 'objectives', 'variables']
    enc = OneHotEncoder(handle_unknown='ignore')
    scaler = StandardScaler()

    if feat_sets == None:
        feat_sets = util.get_default_aggregators()

    # obtain the data, do necessary preprocessing
    _, igd_dict, igd_array_regr = util.create_igd_array_and_dict('indicator_data\\igd_values_log.txt')
    labels_regr = util.get_labels_from_file(igd_labels_2, feat_sets)
    dataframe = get_dataframe(igd_array_regr, labels_regr, feat_sets, problem_instances)
    data, selected_features = prepare_data(dataframe, test_problems, scaler, enc, response_variable, load_features=load_models)
    
    if model_dict == None:
        model_dict = get_model_data()

    if configs == None:
        configs = util.get_benchmark_configurations()

    igd_value_sets = []
    config_labels = []
    mse_values = {}

    # loop through the models and run all of them on the data, create performance profile plots of the results
    for model_name, model_data in model_dict.items():

        model, param_grid = model_data
        print("-"*10, model_name, "-"*10)
        
        # run all of the regression models, either with hyperparameter optimization or using existing models
        mse, igd_values = run_full_regression_model(dataframe, igd_dict, test_problems, model, data, scaler, enc, 
                                                    selected_features, param_grid, model_name, problems_to_ignore,
                                                     response_variable, load_file=load_models)
        util.create_performance_profile_plot(igd_dict, igd_values, configs, test_problems, model_name + '_regressor')
        igd_value_sets.append(igd_values)
        config_labels.append(model_name + ' regressor')
        mse_values[model_name] = mse
    
    util.create_performance_profile_plot(igd_dict, igd_value_sets, None, test_problems, 'regressors', config_labels, font_size=6)

    for model, mse_value in mse_values.items():
        print(model, mse_value)

    calculate_r2_scores(feat_sets, response_variable)


if __name__ == "__main__":
    do()
    
    