import pandas as pd
import numpy as np
import time
import pickle

import pycaret
from pycaret import classification as clscaret
from pycaret import regression as regcaret
from pycaret.utils import check_metric
from sklearn.metrics import roc_curve
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split

from omegaconf import OmegaConf
from argparse import ArgumentParser

from pycaret.datasets import get_data



def autoML(config):

    config = config

    modeltype = config.modeltype
    modelname = config.modelname
    n = config.num_models

    if modeltype.lower() == 'classification':

        modeltag = 'clscaret'
        metricgraph = 'aucgraph'
        test_metrics = ['Accuracy', 'AUC', 'F1', 'Precision', 'Recall']  

    elif modeltype.lower() == "regression":
        
        modeltag = "regcaret"
        metricgraph = "errorplot"
        test_metrics = ['MAE', 'MSE', 'RMSE', 'R2', 'RMSLE', 'MAPE']

    

    data = get_data('diamond')

    train_data = data.sample(frac=0.80, random_state=2021)
    test_data = data.drop(train_data.index)
    test_data = test_data[data.columns]


    data = eval(modeltag).setup(data=train_data, target = 'Price', transform_target = True)


    if modelname:
        model = eval(modeltag).create_model(modelname, round=2)

        if modeltag == "regcaret":
            unseen_predictions = eval(modeltag).predict_model(model, data=test_data)
        else:
            unseen_predictions = eval(modeltag).predict_model(model, data=test_data, raw_score=True)

        eval(modeltag).save_model(model, 'xyz')

        metrics = {"models": [modelname], "type": modeltype,
                   "modelfile": str(modelname) + "_" + str(modeltype)}

        for i in test_metrics:
            metrics[i] = check_metric(unseen_predictions.eval("Price"), unseen_predictions.Label, i)
        

    else:
        topmodels = eval(modeltag).compare_models(n_select=n)

        topmodelnames = list(eval(modeltag).pull()['Model'])

        final_model = topmodels[0]
        final_modelname = list(eval(modeltag).pull()['Model'])[0]


        if modeltag == "regcaret":
            unseen_predictions = eval(modeltag).predict_model(final_model, data=test_data)
        else:
            unseen_predictions = eval(modeltag).predict_model(final_model, data=test_data, raw_score=True)

        eval(modeltag).save_model(final_model, 'xyz')

        metrics = {"models": [final_modelname], "type": modeltype,
                   "modelfile": str(final_modelname) + "_" + str(modeltype)}

        for i in test_metrics:
            metrics[i] = check_metric(unseen_predictions.eval("Price"), unseen_predictions.Label, i)


    return metrics



if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="provide the config file")
    args = parser.parse_args()

    # get configuration
    configFile = OmegaConf.load(args.config)
    config = configFile.config

    finalmetrics = autoML(config)
    print(finalmetrics)
