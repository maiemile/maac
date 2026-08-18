# code by @maiemile

import utils as util
from generate_database import query_data, get_best_configs_dictionary, get_eas_dictionary

import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os
from pathlib import Path

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score, make_scorer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Fetch the information on whether to load pre-existing models (True) or train new ones (False)
load_models = bool(util.load_param_config('load_models'))
# Load the filename of the database and the base path
database = util.load_param_config('database_file')

def get_model_data() -> dict:
    '''
    Returns a dictionary of the default machine learning models and their parameter grids for hyperparameter optimization.
    '''

    # hyperparameter optimization for the machine learning models => split into train/val + test sets
    # and evaluate the best model with the test set to get a more accurate representation of the accuracy
    clf_dt = MultiOutputClassifier(DecisionTreeClassifier(random_state=42))
    clf_rf = MultiOutputClassifier(RandomForestClassifier(random_state=42))
    clf_lr = MultiOutputClassifier(LogisticRegression(random_state=42))
    clf_xg = MultiOutputClassifier(xgb.XGBClassifier(random_state=42))
    clf_nn = MultiOutputClassifier(MLPClassifier(random_state=42, max_iter=500))

    param_grid_rf = {
        "estimator__n_estimators": [10,50,100,200],
        "estimator__criterion": ["gini", "entropy", "log_loss"],
        "estimator__max_depth": [None, 2,4,7],
        "estimator__max_features": [None, "sqrt", "log2"],
    }
    param_grid_dt = {
        "estimator__criterion": ["gini", "entropy", "log_loss"],
        "estimator__max_depth": [None, 3,5,10],
        "estimator__max_features": [None, "sqrt", "log2"],
        "estimator__splitter": ["best", "random"]
    }
    param_grid_lr = {
        "estimator__l1_ratio": [0, 0.25, 0.5, 0.75, 1],
        "estimator__solver": ['lbfgs','sag', 'saga']
    }
    param_grid_xg = {
        "estimator__max_depth": [6,8,10,12],
        "estimator__subsample": [0.5, 0.75, 1],
        "estimator__eta": [0.01, 0.1, 0.3, 0.6],
        "estimator__n_estimators": [10,50,100,200],
    }
    param_grid_nn = {
        "estimator__hidden_layer_sizes": [(30,10,6), (20,12,4), (50, 30, 10, 4), (16,6), (12,4)],
        "estimator__solver": ["adam", "lbfgs"],
        "estimator__learning_rate": ["constant", "adaptive"],
        "estimator__activation": ["logistic", "relu"]
    }

    model_dict = {
        "Random forest": [clf_rf, param_grid_rf], 
        "Decision tree": [clf_dt, param_grid_dt], 
        "Logistic regression": [clf_lr, param_grid_lr],
        #"XGBoost": [clf_xg, param_grid_xg], # TODO: there is a persistent issue with XGBoost when some classes are rare
        "Neural network": [clf_nn, param_grid_nn]
        }
    
    return model_dict


def multioutput_macro_f1(y_true, y_pred) -> float:
    '''
    Implement a multioutput Macro F1 score function.
    Macro F1 score is calculated separately for each output,
    and arithmetic mean is taken from these scores to get the final output.
    '''
    # y_true, y_pred are arrays of shape (n_samples, n_outputs)
    per_output = []
    for i in range(y_true.shape[1]):
        # Use 'macro' to treat all classes equally per output
        per_output.append(f1_score(y_true.iloc[:, i], y_pred[:, i], average='macro', zero_division=0))
    return float(np.mean(per_output))


def select_features(X_train, y_train) -> list[str]:
    '''
    Does feature selection according to feature importance.
    20 most important features are selected for each response variable
    and the union of all features are used to obtain a subset of X_train
    and X_test.
    '''

    if load_models:
        # Try to find a classifier model in the models folder
        modelname = None
        for x in os.listdir('models'):
            # TODO: could allow other file types
            if x.endswith(".pkl"):
                if '_classifier' in x:
                    modelname = x
                    break
        
        # If no classifier models were found, raise an exception
        if modelname == None:
            raise Exception('No models found. Feature names could not be loaded.')
        
        # the following code is used when wanting to access the features used for a model
        with open(Path(f'models/{modelname}'), 'rb') as f:
            clf2 = pickle.load(f)
        for clf in clf2.estimators_:
            features = clf.feature_names_in_
            break

        return features

    else:
        selected_features = []

        # Select 20 most important features for each response variable separately
        # and concatenate the chosen features
        for i in range(len(y_train.columns)):
            selector = SelectKBest(f_classif, k=20)
            X_new = selector.fit_transform(X_train, y_train.iloc[:,i])

            # print a dataframe of the selected features, ordered by the score
            names = X_train.columns.values[selector.get_support()]
            scores = selector.scores_[selector.get_support()]
            names_scores = list(zip(names, scores))
            ns_df = pd.DataFrame(data = names_scores, columns=
             ['Feat_names','F_Scores'])
            ns_df_sorted = ns_df.sort_values(['F_Scores','Feat_names'], ascending =
             [False, True])
            #print(ns_df_sorted)

            columns = list(selector.get_feature_names_out())

            selected_features = selected_features + columns

        # remove duplicates
        union_list = list(set(selected_features))

        return union_list


def prepare_data(df, test_problems, scaler, indicator):
    '''
    Does data preprocessing. Missing and unrealistic values are handled. 
    Categorical variables are encoded, numerical variables are scaled and train/test split is performed.
    Features are selected based on importance or loaded from existing models.
    '''

    # Remove all columns with missing or infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=1)

    y_cols = util.get_param_names()

    # scale or encode train data depending on if the feature is numerical or categorical
    categorical_vars = df[y_cols]
    cols = []
    encs = []
    for col in y_cols:
        enc = LabelEncoder().fit(categorical_vars[col])
        encs.append(enc)

        y1_enc = enc.transform(categorical_vars[col])
        cols.append(y1_enc)
    y = pd.DataFrame(np.column_stack(cols), columns=y_cols)

    # replace the response variable columns with the encoded values
    df = df.drop(y_cols, axis=1)
    df = pd.concat([df, y], axis=1)
    print(df.head())

    # train/test split
    df_test = df[df['problem_id'].isin(test_problems)]
    df = df.drop(df[df['problem_id'].isin(test_problems)].index)

    # remove unnecessary features
    try:
        problem_data = ['problem_id', 'seed', 'ea_id', 'name', indicator]
        df = df.drop(problem_data, axis=1)
    except:
        problem_data = ['problem_id', 'seed', 'ea_id', indicator]
        df = df.drop(problem_data, axis=1)  

    # save the data related to the run in a separate dataframe
    df_test_data = df_test[problem_data]

    # reset the index to be in line with the X_test dataframe
    df_test_data.reset_index(drop=True, inplace=True)
    df_test = df_test.drop(problem_data, axis=1)

    y_train = df[y_cols]
    df_to_normalize = df.drop(columns=y_cols)
    # fix binary columns TODO: change the format of these columns in the database / store them as floats in ELA calculations
    df_to_normalize = util.convert_data(df_to_normalize)
    X_train = scaler.fit_transform(df_to_normalize)

    y_test = df_test[y_cols]
    df_to_normalize = df_test.drop(columns=y_cols)
    # fix binary columns 
    df_to_normalize = util.convert_data(df_to_normalize)
    X_test = scaler.transform(df_to_normalize)    

    column_names = list(df_to_normalize)
    X_train = pd.DataFrame(X_train, columns=column_names)
    X_test = pd.DataFrame(X_test, columns=column_names)

    selected_features = select_features(X_train, y_train)

    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    # TODO: warning when one of the labels is missing from the y_train data

    return [X_train, X_test, y_train, y_test], df_test_data, encs


def train_models(model:str, model_data, scorer, data):
    '''
    Train the hyperparameters of the given model using cross-validation.
    Calculates some relevant metrics and returns the predictions for further analysis.
    '''

    X_train, X_test, y_train, y_test = data
    if load_models:
        #load the model
        with open(Path(f'models/{model}_classifier.pkl'), 'rb') as f:
            clf2 = pickle.load(f)
        best_estimator = clf2
    else:
        print("="*10 + model + "="*10)
        classifier = model_data[0]
        param_grid = model_data[1]

        # Use cross-validation as the dataset is small
        # TODO: could stratify this?
        kfold = KFold(n_splits=10, shuffle=True, random_state=42)

        # grid search is fine with a small parameter grid
        # TODO: allow other types of cross-validation?
        grid_search = GridSearchCV(
            estimator=classifier,
            param_grid=param_grid,
            cv=kfold,
            scoring=scorer,   # macro F1 averaged across folds
            verbose=2
        )
        grid_search.fit(X_train, y_train)
        print("Best parameters:", grid_search.best_params_)
        print("Best CV score (macro F1):", grid_search.best_score_)

        #save the model
        with open(Path(f'models/{model}_classifier.pkl'),'wb') as f:
            pickle.dump(grid_search.best_estimator_,f)

        best_estimator = grid_search.best_estimator_

    print("="*10 + model + "="*10)
    
    # Use the best model based on grid search to predict the test set response variables
    y_pred_test = best_estimator.predict(X_test)

    # Calculate the metrics
    per_output_f1 = [f1_score(y_test.iloc[:, i], y_pred_test[:, i], average='macro', zero_division=0)
                     for i in range(y_test.shape[1])]
    # Compute per-output F1 (weighted to account for class imbalance, especially in algorithm selection)
    f1_per_output_w = [f1_score(y_test.iloc[:, i], y_pred_test[:, i], average='weighted') for i in range(y_test.shape[1])]
    macro_avg_f1 = np.mean(per_output_f1)
    weighted_avg_f1 = np.mean(f1_per_output_w)
    exact_match = np.mean(np.all(y_pred_test == y_test, axis=1))

    # Print the results
    print("Test per-output macro F1:", np.round(per_output_f1, 3))
    print("Test per-output weighted F1:", np.round(f1_per_output_w, 3))
    print("Test macro average F1:", np.round(macro_avg_f1, 3))
    print("Test weighted average F1:", np.round(weighted_avg_f1, 3))
    print("Test exact match accuracy:", np.round(exact_match, 3))

    return y_pred_test


def get_predicted_labels(y_pred_test, enc):
    '''
    Performs inverse transform on the predictions to obtain true predicted labels.
    '''
    # reverse transform the predicted labels into strings
    y_pred_test_tf = []
    for i in range(len(enc)):
        y_pred_test_tf.append(enc[i].inverse_transform(y_pred_test[:,i]))
    
    y_pred_test_df = pd.DataFrame(np.column_stack(y_pred_test_tf))

    return y_pred_test_df


def get_predicted_igd(test_problems: list[int], problem_data, y_pred, model_name:str, indicator:str, single_best_solver:int) -> list[float]:
    '''
    Returns the IGD values of the configurations predicted by the model.
    '''

    # create a dataframe with the optimal config and the predictions
    params = util.get_param_names()
    y_pred_df = pd.DataFrame(np.array(y_pred), columns=params)
    new_df = pd.concat([problem_data, y_pred_df], axis=1)

    optimal_configs_test = []
    predicted_configs_test = []
    igd_values = []
    sbs_igd_values = []

    for problem in test_problems:
        seed = 1
        while True:
            # get the best config + prediction for the given problem
            best_config = new_df.loc[(new_df['problem_id'] == problem) & (new_df['seed'] == seed)]

            # only continue the loop if there are results for the given problem + seed pair
            if len(best_config) < 1:
                break

            # get the EA id of the best config
            ea_id = best_config["ea_id"].item()

            # get the parameters of the best config
            sql = f'''SELECT {','.join(params)} FROM eas WHERE ea_id = {ea_id}'''
            res_ea = query_data(sql)

            # append the best configuration to a list
            optimal_configs_test.append(list(res_ea))

            # get the prediction from the data and append it to a list
            config = best_config[params].values.tolist()[0]
            predicted_configs_test.append(config)

            # get the EA id of the predicted configuration
            sql = '''SELECT ea_id FROM eas WHERE'''
            for param, val in zip(params, config):
                sql += f''' {param} = '{val}' AND'''
            sql = sql[:-3]
            pred_ea_id = query_data(sql)

            # get the IGD value of the predicted configuration
            sql = f'''SELECT {indicator} FROM runs WHERE ea_id = {pred_ea_id[0]} AND problem_id = {problem}'''
            res = query_data(sql)
            # TODO: temporary solution to handle None values
            # RVEA-NUM must be handled/ignored properly...
            for i in range(len(res)):
                if res[i][0] == None:
                    res[i] = (9999999,)
 
            median_igd = np.median(np.array(res))
            igd_values.append(median_igd)

            # get the IGD value of the SBS
            sql = f'''SELECT {indicator} FROM runs WHERE ea_id = {single_best_solver} AND problem_id = {problem}'''
            res = query_data(sql)
            median_igd = np.median(np.array(res))
            sbs_igd_values.append(median_igd)
            seed += 1

    # create and save confusion matrices of the predicted parameters of the configurations
    util.create_confusion_matrices(np.asarray(optimal_configs_test), np.asarray(predicted_configs_test), model_name + '_classifier', params)

    return igd_values, sbs_igd_values


def do(model_dict: dict = None, configs: list[str] = None, indicator: str = None) -> None:
    '''
    The default function for running the full pipeline of classification-based configurator models.
    '''

    if model_dict == None:
        model_dict = get_model_data()

    if indicator == None:
        # Load the default indicator
        indicator = util.load_param_config('indicator')

    # fetch the ids of problems used in the testing phase
    test_prob = ["dtlz3", "wfg7", "re31", "re32", "re33", "re34", "re37", "re41", "re42", "re61"] 
    test_problems = util.get_test_problems(test_prob)

    scaler = StandardScaler()

    # Custom scoring function based on Macro F1 score
    scorer = make_scorer(multioutput_macro_f1)

    # Combine data from runs and features tables into one pandas dataframe
    try:  
        con = sqlite3.connect(database)
        sql = f"""SELECT * FROM features"""
        df = pd.read_sql_query(sql, con)
        problem_ids = df['problem_id']

        # get the best configuration by problem
        best_configs = get_best_configs_dictionary(indicator)
        data_best = []
        for id in problem_ids:
            data_best.append(best_configs[id])

        # combine the feature and EA data
        data_best_df = pd.DataFrame(data_best, columns=['ea_id', indicator])
        new_df = pd.concat([df, data_best_df], axis=1)

        ea_dict = get_eas_dictionary()

        ea_data = []
        ea_ids = new_df["ea_id"]
        for id in ea_ids:
            ea_data.append(ea_dict[id])

        y_cols = util.get_param_names()
        data_ea_df = pd.DataFrame(ea_data, columns=["name"] + y_cols)

        df = pd.concat([new_df, data_ea_df], axis=1)
    except sqlite3.Error as e:
        print(e)
    finally:
        con.close()

    # load the single best solver, ignoring the test problems
    single_best_solver = util.determine_single_best_solver(test_problems=test_problems)

    # perform data preprocessing
    data, problem_data, encs = prepare_data(df, test_problems, scaler, indicator)

    igd_value_sets = []
    config_labels = []

    # loop through all models, evaluate the predictions and create plots
    for model_name, model_data in model_dict.items():
        # train the model, return the predictions
        y_pred_test = train_models(model_name, model_data, scorer, data)

        # inverse transform the predictions back to original values
        y_pred_test_df = get_predicted_labels(y_pred_test, encs)

        # get the IGD values achieved by the predictions and the SBS
        igd_values, sbs_igd_values = get_predicted_igd(test_problems, problem_data, y_pred_test_df, model_name, indicator, single_best_solver)
        config_results = [sbs_igd_values, igd_values]

        # convert results to a dataframe
        configs = ['SBS', model_name + ' classifier']
        igd_df = pd.DataFrame(np.array(config_results).T, columns=configs)
        
        # display a proper performance profile plot comparing the configurator against the single best solver
        util.create_performance_profile_plot(igd_df, configs, model_name + '_classifier')

        igd_value_sets.append(igd_values)
        config_labels.append(model_name + ' classifier')

    # performance profile plot for comparing the configurators against each other
    igd_df = pd.DataFrame(np.array(igd_value_sets).T, columns=config_labels)
    util.create_performance_profile_plot(igd_df, config_labels, 'classifiers', font_size=6)   

    # TODO: currently print_decision_trees isn't called


if __name__ == "__main__":
    do()