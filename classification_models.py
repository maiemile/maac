# code by @maiemile

import pickle
import os
import numpy as np
import pandas as pd
import utils as util
import xgboost as xgb
import matplotlib.pyplot as plt
from pathlib import Path
import scienceplots
plt.style.use(['science','no-latex'])

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import cross_val_score, KFold, train_test_split, GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix, make_scorer, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier

# all the problems instances
problem_instances = util.get_problem_instances()
test_problems = util.get_test_problems()

# Fetch the information on whether to load pre-existing models (True) or train new ones (False)
load_models = util.load_files_config()


def load_response_variables(problems_to_ignore):
    Y = []

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
            Y.append([problem] + split_config)

    return Y


def preprocess_data(df_original):
    # Split the data into X and y (input and response variables)
    y_cols_original = ['algo', 'crossover', 'mutation']
    y = df_original[y_cols_original]
    y_cols = y_cols_original + ['problem', 'ic.eps_ratio_MIN', 'ic.eps_ratio_AVG', 'ic.eps_ratio_SD']

    # Drop the following column because it contains several -infinite features
    df = df_original.drop(columns=y_cols)

    # LabelEncoder is fine for converting the response variables in classification models
    # even if they are categorical and unordinal
    # each output requires a separate encoder as they have different value sets
    cols = []
    encs = []
    for col in y_cols_original:
        enc = LabelEncoder().fit(y[col])
        encs.append(enc)

        y1_enc = enc.transform(y[col])
        cols.append(y1_enc)

    y = pd.DataFrame(np.column_stack(cols))

    # print all columns with null values
    nan_cols = [i for i in df.columns if df[i].isnull().any()]
    for value in nan_cols:
        print(value)

    # dataset contains no nan values
    print(df.isnull().values.any())

    # infinite values have been removed
    infs = (df == -np.inf).any(axis=0)
    for i in range(len(infs)):
        if infs[i] == True:
            print(i, infs[i])

    infs = (df == np.inf).any(axis=0)
    for i in range(len(infs)):
        if infs[i] == True:
            print(i, infs[i])

    # Scale the input data to follow Gaussian distribution with a mean of 0 and standard deviation of 1
    scaler = StandardScaler()
    column_names = list(df)
    X_temp = scaler.fit_transform(df)
    X_temp = pd.DataFrame(X_temp, columns=column_names)

    # split the data into train and test sets based on a manually picked test problem set
    test_indexes = df_original[df_original['problem'].str.contains('|'.join(test_problems), na=False)].index

    X_train = X_temp.drop(test_indexes)
    X_test = X_temp.iloc[test_indexes]
    y_train = y.drop(test_indexes)
    y_test = y.iloc[test_indexes]

    return X_train, X_test, y_train, y_test, y, encs


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


def select_features(y, X_train, X_test, y_train):
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
                # TODO: is _classifier sufficient as an identifier of classification models
                if '_classifier' in x:
                    modelname = x
                    break
        
        # If no classifier models were found, raise an exception
        if modelname == None:
            raise Exception('No models found. Feature names could not be loaded.')
        
        # the following code is used when wanting to access the features used for a model
        with open(f'models\{modelname}', 'rb') as f:
            clf2 = pickle.load(f)
        for clf in clf2.estimators_:
            features = clf.feature_names_in_
            break

        X_train = X_train[features]
        X_test = X_test[features]

    else:
        selected_features = []
        selected_feature_indexes = []

        # Select 20 most important features for each response variable separately
        # and concatenate the chosen features
        for i in range(len(y.columns)):
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
            print(ns_df_sorted)

            cols_idxs = list(selector.get_support(indices=True))
            columns = list(selector.get_feature_names_out())

            selected_features = selected_features + columns
            selected_feature_indexes = selected_feature_indexes + cols_idxs
            print(cols_idxs)
            print(columns)

        # remove duplicates
        union_list = list(set(selected_features))
        union_list_idx = list(set(selected_feature_indexes))

        print(union_list)
        print(union_list_idx)
        print(len(union_list))
        print(len(union_list_idx))
        X_train = X_train[union_list]
        X_test = X_test[union_list]

    return X_train, X_test


def get_model_data() -> dict:
    '''
    Returns a dictionary of the default machine learning models and their parameter grids for hyperparameter optimization
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
        "XGBoost": [clf_xg, param_grid_xg],
        "Neural network": [clf_nn, param_grid_nn]
        }
    
    return model_dict


def train_models(model_dict, scorer, X_train, y_train):

    best_estimators = {}
    best_params = {}
    for model, model_data in model_dict.items():
        if load_models:
            #load the model
            with open(f'models\\{model}_classifier_v2.pkl', 'rb') as f:
                clf2 = pickle.load(f)

            best_estimators[model] = clf2

        else:
            print("="*10 + model + "="*10)
            classifier = model_data[0]
            param_grid = model_data[1]

            # Use cross-validation as the dataset is small
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)

            grid_search = GridSearchCV(
                estimator=classifier,
                param_grid=param_grid,
                cv=kfold,
                scoring=scorer,   # macro F1 averaged across folds
                verbose=1
            )
            grid_search.fit(X_train, y_train)

            print("Best parameters:", grid_search.best_params_)
            print("Best CV score (macro F1):", grid_search.best_score_)

            #save the model
            with open(f'models\\{model}_classifier_v2.pkl','wb') as f:
                pickle.dump(grid_search.best_estimator_,f)

            best_estimators[model] = grid_search.best_estimator_
            best_params[model] = grid_search.best_params_

    return best_estimators


def evaluate_models(model_name, best_model, enc, X_test, y_test, y):
    print("="*10 + model_name + "="*10)
    
    # Use the best model based on grid search to predict the test set response variables
    y_pred_test = best_model.predict(X_test)

    # Calculate the metrics
    per_output_f1 = [f1_score(y_test.iloc[:, i], y_pred_test[:, i], average='macro', zero_division=0)
                     for i in range(y.shape[1])]
    # Compute per-output F1 (weighted to account for class imbalance, especially in algorithm selection)
    f1_per_output_w = [f1_score(y_test.iloc[:, i], y_pred_test[:, i], average='weighted') for i in range(y.shape[1])]
    macro_avg_f1 = np.mean(per_output_f1)
    weighted_avg_f1 = np.mean(f1_per_output_w)
    exact_match = np.mean(np.all(y_pred_test == y_test, axis=1))

    # Print the results
    print("Test per-output macro F1:", np.round(per_output_f1, 3))
    print("Test per-output weighted F1:", np.round(f1_per_output_w, 3))
    print("Test macro average F1:", np.round(macro_avg_f1, 3))
    print("Test weighted average F1:", np.round(weighted_avg_f1, 3))
    print("Test exact match accuracy:", np.round(exact_match, 3))

    # Confusion matrix or each output
    encs = enc
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,4))
    axes = [ax1, ax2, ax3]
    for j in range(y.shape[1]):
        cm = confusion_matrix(y_test.iloc[:, j], y_pred_test[:, j])
        print(cm)
        # A "dumb" way to deal with rvea not appearing in the test set or predictions
        # TODO: needs to be improved to handle missing predictions
        if j != 0:
            if j == 1:
                labels = [r"BLX-$\alpha$", "LX", "SBX", "SAX"]
            else:
                labels = encs[j].inverse_transform(best_model.classes_[j])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=labels)
        else:
            y_pred_test_enc = enc[0].inverse_transform(y_pred_test[:,0])
            if "rvea" in y_pred_test_enc:
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

    plt.savefig(f'figures\\confusion_matrices\\{model_name}_v2.pdf')
    plt.show()

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
    
    print(y_pred_test_df)

    return y_pred_test_df


def get_predicted_igd_values(y_pred_test_df, y_test, df_original, indexes, igd_dict):
    problems = []
    problems_and_configs = []
    igd_values = []
    
    # for each test problem, calculate the IGD value achieved by the predicted configuration
    for i in range(len(y_test.index)):
        problems.append(df_original.iloc[indexes[i]]["problem"])
        row =  y_pred_test_df.iloc[i]
        problem_plus_config = df_original.iloc[indexes[i]]["problem"] + '-' + row[0] + '-' + row[1] + '-' + row[2]
        problems_and_configs.append(problem_plus_config)
        igd_values.append(igd_dict[problem_plus_config][0])

    print(problems)
    print(igd_values)

    return igd_values, problems


def print_decision_trees() -> None:
    from sklearn import tree
    from matplotlib.colors import ListedColormap, to_rgb
    model = "Decision tree"
    classes_full = [
        ["IBEA", "NSGA-III", "RVEA"],
        ["BLX-a", "LX", "SBX", "SAX"],
        ["BPM", "MPTM", "NUM", "PM"]
    ]
    colors = ['orange', 'lightblue', 'plum', 'indianred']
    with open(f'models\\{model}_classifier.pkl', 'rb') as f:
        clf2 = pickle.load(f)
        for i in range(len(clf2.estimators_)):
            clf = clf2.estimators_[i]
            features = clf.feature_names_in_
            classes = classes_full[i]
            plt.figure(figsize=(16,12))
            treeplots = tree.plot_tree(clf, fontsize=11, feature_names=features, class_names=classes, impurity=False, filled=True, rounded=True)
            treeplots_fixed = []
            for treeplot in treeplots:
                text = treeplot.get_text() 
                if not any(x in text for x in ["True", "False"]):
                    treeplots_fixed.append(treeplot)

            for treeplot, impurity, value in zip(treeplots_fixed, clf.tree_.impurity, clf.tree_.value):
                # let the max value decide the color; whiten the color depending on impurity (gini)
                r, g, b = to_rgb(colors[np.argmax(value)])
                f = impurity * len(classes)/(len(classes)-1) # for N colors: f = impurity * N/(N-1) if N>1 else 0
                try:
                    treeplot.get_bbox_patch().set_facecolor((f + (1-f)*r, f + (1-f)*g, f + (1-f)*b))
                    treeplot.get_bbox_patch().set_edgecolor('black')
                except:
                    continue
            plt.show()


def do(model_dict: dict = None, feat_sets: list[str] = None, configs: list[str] = None, problems_to_ignore: list[str] = []) -> None:

    igd_array, igd_dict, _ = util.create_igd_array_and_dict('indicator_data\\igd_values_log.txt')

    Y = load_response_variables(problems_to_ignore)

    labels = ['problem', 'algo', 'crossover', 'mutation', 'objectives', 'variables']

    if feat_sets == None:
        feat_sets = util.get_default_aggregators()

    labels = util.get_labels_from_file(labels, feat_sets)
    data = util.load_data(Y, feat_sets, problem_instances)

    # create a Pandas dataframe
    df_original = pd.DataFrame(data, columns=labels)

    X_train, X_test, y_train, y_test, y, enc = preprocess_data(df_original)
    #print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    indexes = list(y_test.index)

    # Select most relevant features and update the input variable dataset to reflect the selections
    X_train, X_test = select_features(y, X_train, X_test, y_train)

    # Custom scoring function based on Macro F1 score
    scorer = make_scorer(multioutput_macro_f1)
    if model_dict == None:
        model_dict = get_model_data()
    best_estimators = train_models(model_dict, scorer, X_train, y_train)

    if configs == None:
        configs = util.get_benchmark_configurations()

    igd_value_sets = []
    config_labels = []
    for model_name, best_model in best_estimators.items():
        y_pred_test = evaluate_models(model_name, best_model, enc, X_test, y_test, y)
        y_pred_test_df = get_predicted_labels(y_pred_test, enc)
        igd_values, testproblems = get_predicted_igd_values(y_pred_test_df, y_test, df_original, indexes, igd_dict)

        # display a proper performance profile plot comparing the configurator against the above configurations
        util.create_performance_profile_plot(igd_dict, igd_values, configs, testproblems, model_name + '_classifier')

        igd_value_sets.append(igd_values)
        config_labels.append(model_name + ' classifier')

    util.create_performance_profile_plot(igd_dict, igd_value_sets, None, test_problems, 'classifiers', config_labels, font_size=6)   

    # TODO: currently print_decision_trees isn't called


if __name__ == "__main__":
    do()