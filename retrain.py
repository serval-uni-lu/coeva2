import copy, json, sys

import pandas as pd
import numpy as np
from joblib import load
import logging
from pathlib import Path
from pymoo.optimize import minimize
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn import metrics
import sys, getopt, os

sys.path.append("./")
from src.coeva2.venus_encoder import VenusEncoder
from src.coeva2.venus_attack_generator import init_attack
from src.coeva2.problem_definition import ProblemConstraints, ProblemEvaluation

from src.utils.in_out import json_to_file, save_to_file, pickle_from_file, load_from_file


def run(config_file, experiment_id=None, nb_retrain=3000):

    parameters= {}
    with open(config_file) as f:
        parameters = json.load(f)

    if parameters.get("experiment_path", None) is None:
        raise KeyError()

    
    experiment_id =  int(datetime.timestamp(datetime.now())) if experiment_id is None else experiment_id
    dataset_path = parameters.get("dataset_path")
    dataset_features = parameters.get("dataset_features")
    experiment_path = parameters.get("experiment_path")
    scaler_file = parameters.get("scaler_file")
    model_file = parameters.get("model_file")
    threshold = parameters.get("model_threshold")
    ga_parameters = parameters.get("ga_parameters")
    dataset_constraints = parameters.get("dataset_constraints")
    max_states = parameters.get("max_states",0)
    initial_states = parameters.get("initial_states",0)
    
    print("running config {} experiment {} id {}".format(config_file, experiment_path, experiment_id))

    data = pd.read_csv(dataset_path)
    model = load("{}/{}".format(experiment_path,model_file))
    scaler = pickle_from_file("{}/{}".format(experiment_path,scaler_file))
    encoder = VenusEncoder(dataset_features, len(ga_parameters.get("gene_types")))
    problem_constraints = ProblemConstraints()


    y = data.pop("charged_off").to_numpy()
    X = data.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
    
    y_pred_proba = model.predict_proba(X_test)
    y_pred = (y_pred_proba[:, 1] >= threshold).astype(bool)
    accuracy_before = metrics.accuracy_score(y_test,y_pred)


    initial_X  = load_from_file("{}/{}".format(experiment_path,initial_states))

    print("initial attack")
    valid_adversarials = []
    for i,state in enumerate(initial_X):
        
        if (max_states>0 and i > max_states) or len(valid_adversarials) > nb_retrain:
            break

        initial_state = copy.copy(state)
        problem, algorithm, termination = init_attack(
                state, model, scaler, encoder, problem_constraints, **ga_parameters
        )
        result = minimize(
            problem, algorithm, termination, verbose=2, save_history=False,
        )

        evaluation = ProblemEvaluation(result, encoder, initial_state, threshold, model)
        valid, objectives = evaluation.calculate_objectives(True)

        if valid is not None and len(valid)>0:
            valid_adversarials = valid + valid_adversarials


    # all the new elements should be rejected, i.e prediction = 1
    X_train = np.append(X_train, [valid_adversarials], axis=0)
    y_train = np.append(y_train, [np.ones(max_states)], axis=0)

    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)
    y_pred = (y_pred_proba[:, 1] >= threshold).astype(bool)
    accuracy_after = metrics.accuracy_score(y_test,y_pred)

    
    output_dir = "{}/states/{}_retrain".format(experiment_path,experiment_id)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_objectives = []
    for i,state in enumerate(initial_X[max_states:]):
        
        if (max_states>0 and i > max_states) or len(valid_adversarials) > nb_retrain:
            break

        initial_state = copy.copy(state)
        problem, algorithm, termination = init_attack(
                state, model, scaler, encoder, problem_constraints, **ga_parameters
        )
        result = minimize(
            problem, algorithm, termination, verbose=2, save_history=False,
        )

        evaluation = ProblemEvaluation(result, encoder, initial_state, threshold, model)
        objectives = evaluation.calculate_objectives()
        all_objectives.append(objectives)

        save_to_file(objectives,"{}/s{}.npy".format(output_dir,i))


    df = pd.DataFrame(data=all_objectives, columns=["respectsConstraints", "isMisclassified", "o3", "o4"])
    print(df)
    print("Accuracy before adversarial training:{} after: {}".format(accuracy_before, accuracy_after))
    

def init(argv):

    config_file  = "./configurations/config_f1f2_fast.json"
    experiment_id = "f1f2f4_retrain"
    nb_retrains = 3000

    try:
        opts, args = getopt.getopt(argv, "hc:n:i:", [
                                   "config=","nb=", "id="])
    except getopt.GetoptError:
        pass
    for opt, arg in opts:
        
        if opt == '-h':
            print(
                'retrain.py -c <json configuration> [-i <run id> -nn <nb adversarial to use in retrain>]')
            sys.exit()
        elif opt in ("-c", "--config"):
            config_file = arg

        elif opt in ("-i", "--id"):
            experiment_id = arg

        elif opt in ("-n", "--nb"):
            nb_retrains = int(arg)


    if config_file is None:
        print(
            'Please provide a configuration file . At least: "retrain.py -c <json configuration>"')
        sys.exit()
        
    run(config_file, experiment_id, nb_retrains)


if __name__ == "__main__":
    init(sys.argv[1:])