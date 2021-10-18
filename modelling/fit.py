import pandas as pd
import numpy as np

#Lista de modelos
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

#Comparacao de modelos
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, fbeta_score

from joblib import dump, load


def fit_models(X_train: pd.DataFrame, 
               y_train: pd.DataFrame, 
               X_test: pd.DataFrame, 
               y_test: pd.DataFrame) -> pd.DataFrame:
    '''
    Lightweight script to test many models and find winners
    :param X_train: training split
    :param y_train: training target vector
    :param X_test: test split
    :param y_test: test target vector
    :return: DataFrame of predictions
    '''
    
    metricas = [('auc',roc_auc_score),
            ('bal_accuracy',balanced_accuracy_score),
            ('f1_score',f1_score),
            ('f2_score',fbeta_score)
           ]


    hipers = {'LogClass':load('Hiperparametrizados/Logi.joblib'),
                'RF':load('Hiperparametrizados/Rand.joblib'),
                'MLP':load('Hiperparametrizados/MLPC.joblib'),
                'SVM':load('Hiperparametrizados/SVC(.joblib'),
                'XGB':load('Hiperparametrizados/XGBC.joblib')}

    modelos = [LogisticRegression(),
               RandomForestClassifier(),
               MLPClassifier(),
               SVC(),
               XGBClassifier()]

    for i,j in zip(hipers, modelos):
        hipers[i] = j.set_params(**hipers[i].best_params_)


    # Criando listas para armazenamento
    AUC = []
    bal_acc = []
    f1 = []
    f2 = []
    modelo = []

    for m in hipers:
        clf = hipers[m].fit(X_train, y_train) #ajuste
        y_pred = clf.predict(X_test) #predicao

        modelo.append(m)
        
        if y_test.sum() == 0:
            AUC.append(np.nan)
            bal_acc.append(np.nan)
            f1.append(np.nan)
            f2.append(np.nan)
        else:
            AUC.append(roc_auc_score(y_test, y_pred))
            bal_acc.append(balanced_accuracy_score(y_test, y_pred))
            f1.append(f1_score(y_test, y_pred))
            f2.append(fbeta_score(y_test, y_pred, beta=2))

    df = pd.DataFrame({'AUC':AUC,
                        'B_ACC':bal_acc,
                        'F1':f1,
                        'F2':f2,
                        'Model':modelo
                            })
    return df