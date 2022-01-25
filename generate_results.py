from pyclbr import Function
import random
from collections import defaultdict
from typing import Dict, Iterable, Tuple, List
from tqdm import tqdm
from pprint import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score


TSV_LOCATION = "../Random-forest-data-v2.tsv"
NUM_TRIALS = 20
SAVE_LOCATION = "./"


def collect_data(include_post_tumor_data: bool=False) -> Tuple[pd.DataFrame, list]:
    """ Collects the data from disk and then """
    full_data = pd.read_csv(TSV_LOCATION, sep="\t")
    # Turn gender from strings into binary data
    full_data.Gender = [1 if x == "M" else 0 for x in full_data['Gender']]
    # These patient IDs are the ones that had the wrong amount of radiation
    # remove them at the start
    patients_to_remove = ["01-001", "01-002", "01-003"]
    final_data = full_data[[(x not in patients_to_remove) for x in full_data["Patient"]]]

    columns_in_X = [
        'tumor_PRE_VECTRA_MHCII_CK',
        'tumor_PRE_VECTRA_PDL1',
        'Gender', 
        'Age',
        'Smoking', 
        'Time_to_surgery', 
        'Radiation_dose',
        'T_stage', 
        'N_stage', 
        'Previous_RT_cat',
        'tumor_PRE_VECTRA_CD8_Treg_ratio', 
        'blood_PRE_TCR', 
        'blood_POST_TCR', 
        'blood_POST_Astrolabe_CD4_effector',
        'blood_PRE_suppressive_myeloids', 
        'blood_post_activated_T_cells',
        'blood_POST_naive_T_cells'
    ]
    
    if include_post_tumor_data:
        columns_in_X += [
            'tumor_POST_VECTRA_MHCII_CK', 
            'tumor_POST_VECTRA_PDL1',
            'tumor_POST_VECTRA_CD8_Treg_ratio'
        ]

    final_data["Radiation_dose"] = [str(x) for x in final_data["Radiation_dose"]]
    X = final_data[columns_in_X]
    assert X.shape[0] == 16
    
    X_data = pd.get_dummies(X)
    dummied_x_columns = X_data.columns

    # Turn the response into binary responder/non-responder
    # Based on the outcomes, 4/5 response means they responded
    # <4 means they did not
    Y_data = [1 if x in [4,5] else 0 for x in final_data.response_by_DOI]
    assert X_data.shape[0] == len(Y_data)

    return X_data, Y_data


def run_seeded_run_with_knn_imputation(
    X: pd.DataFrame, 
    Y: list, 
    rand_seed: int, 
    classifier_fn: Function, 
    grid_to_search: dict
) -> Tuple[float, Dict[str, float]]:
    """ Runs a single seeded run that uses KNN imputation and then trains and evaluates a classifier 
    
    Notes:
        Performs the following steps:
        1) Splits the data into train and validation. There are 16 patients in the data 
            with 75% responders so we split in equal proportions.
        2) Balance the training data into equal numbers of responders and non-responders
        3) Scale the data into range [0, 1]
        4) Impute the training data using KNN imputation
        5) Apply the same scale and imputation method to validation data using the fitted
            transformations from the training data
        6) Perform 3-fold cross validation over the grid on the training data
        7) Return importance values and accuracy on the validation set from the best model
            from the grid search

    """
    random.seed(rand_seed)
    non_responders_in_valid = random.sample([i for i,x in enumerate(Y) if x == 0], k=1)
    responders_in_valid = random.sample([i for i, x in enumerate(Y) if x == 1], k=3)
    validation_slicer = [True if i in non_responders_in_valid + responders_in_valid else False for i in range(X.shape[0])]
    
    validation_X = X[validation_slicer]
    validation_Y = [Y[i] for i in range(16) if validation_slicer[i] is True]

    train_X = X[[not x for x in validation_slicer]]
    train_Y = [Y[i] for i in range(X.shape[0]) if validation_slicer[i] is False]
    assert len(train_Y) == train_X.shape[0]
    assert len(validation_Y) == validation_X.shape[0]
    
    # Upsample the training data such that the model can train on 50/50 responders/non-responders
    non_responders_train_X = train_X[[True if x == 0 else False for x in train_Y]]
    responders_train_X = train_X[[True if x == 1 else False for x in train_Y]]
    upsampled_non_responders_X = resample(non_responders_train_X, n_samples=responders_train_X.shape[0], random_state=rand_seed)
    balanced_train_X = pd.concat([responders_train_X, upsampled_non_responders_X], axis=0)
    balanced_train_Y = [1] * responders_train_X.shape[0] + [0] * upsampled_non_responders_X.shape[0]
    
    # Since the KNN imputer is euclidean distance based we should scale the inputs
    # to be on equal ranges, but only scale based on training data
    min_max_scaler = MinMaxScaler()
    train_data_scaled = min_max_scaler.fit_transform(balanced_train_X)
    
    imputer = KNNImputer(n_neighbors=4)
    imputed_training_data = pd.DataFrame(imputer.fit_transform(train_data_scaled))
    imputed_training_data.columns = train_X.columns
    
    # Then scale and impute the validation data based on the training transformations
    valid_data_scaled = min_max_scaler.transform(validation_X)
    valid_data_imputed = imputer.transform(valid_data_scaled)
    
    
    # Random search of parameters, using 3 fold cross validation
    # search across 100 different combinations, and use all available cores
    classifier = classifier_fn(random_state=rand_seed)
    grid_search = RandomizedSearchCV(
        estimator = classifier, 
        param_distributions = grid_to_search, 
        n_iter = 100, 
        random_state=rand_seed, 
        cv = 3, 
        verbose=0, 
        n_jobs = -1,
        scoring = 'accuracy')
    
    grid_search.fit(imputed_training_data, balanced_train_Y)
    best_mod = grid_search.best_estimator_

    # Get the importance values for the random forest model
    importance_dict = {}
    if type(classifier) is RandomForestClassifier:
        for name, importance in sorted(zip(X.columns, best_mod.feature_importances_), 
                                   key=lambda x: x[1], reverse=True):
            importance_dict[name] = importance
    
    return accuracy_score(best_mod.predict(valid_data_imputed), validation_Y), importance_dict


def repeat_seeded_runs(
    X_data: pd.DataFrame, 
    Y_data: Iterable, 
    classifier_fn: Function, 
    param_grid: Dict[str, list], 
    num_trials: int
) -> Tuple[List[float], Dict[str, float]]:
    """ Runs the seeded run with KNN imputation num_trials times
    
    Args:
        X_data (pd.DataFrame): the input data to predict over. This data 
            will be split into training and validation for each run
        Y_data (Iterable): the outcome data to predict over. Will be split 
            into training and validation for each run
        classifier_fn (function): the sklearn classifier to be used for prediction
        param_grid (dict): a dictionary of parameters to test for the classifier
        num_trials (int): how many times to repeat the experiment
    
    Returns:
        (list): The accuracy for each trial
        (dict): the average importance score per feature over num_trials

    """
    accs = []
    overall_importance_dict = defaultdict(lambda: 0)
    for seed_ind in tqdm(range(0, num_trials)):
        acc, importance_dict = run_seeded_run_with_knn_imputation(X_data, Y_data, seed_ind, classifier_fn, param_grid)
        accs.append(acc)
        for k, v in importance_dict.items():
            overall_importance_dict[k] += v

    averaged_importance_scores = dict()
    for feature, summed_importance in overall_importance_dict.items():
        averaged_importance_scores[feature] = summed_importance/num_trials

    return accs, averaged_importance_scores


def draw_figure(importance_scores, save_location="./"):
    """ Draws and writes the importance score file to disk """
    # Scale the importance score similar to how they display the results in:
    # https://www.nature.com/articles/s41598-021-93162-3
    max_score = max(importance_scores.values())
    scaled_scores = dict()
    for feature, avg_score in importance_scores.items():
        scaled_scores[feature] = avg_score/max_score
    
    sorted_scores = sorted(scaled_scores.items(), key=lambda x: x[1], reverse=True)
    sorted_features = [x[0] for x in sorted_scores]
    normalized_values = [x[1] for x in sorted_scores]

    fig, ax = plt.subplots(figsize=(25, 10))
    y_pos = np.arange(len(scaled_scores))
    ax.barh(y_pos, normalized_values, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_features)
    ax.invert_yaxis()
    ax.set_xlabel('Variable importance')
    
    if save_location:
        plt.savefig(save_location)

    plt.show()
    return


if __name__ == "__main__":
    X, Y = collect_data()
    # Create the random grid
    param_grid = {
        'n_estimators': [int(x) for x in np.linspace(start = 5, stop = 200, num = 20)],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [int(x) for x in np.linspace(3, 10, num = 5)],
        'min_samples_split': [2, 3, 5]
    }
    classifier_fn = RandomForestClassifier

    accuracy_scores, importance_scores = repeat_seeded_runs(X, Y, classifier_fn, param_grid, NUM_TRIALS)
    print(f"Across {NUM_TRIALS} runs the average accuracy score was {np.mean(accuracy_scores)}")

    print("The ranked importance scores were:")
    pprint(sorted(importance_scores.items(), key=lambda x: x[1], reverse=True))

    draw_figure(importance_scores)
