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
from sklearn.metrics import roc_auc_score, f1_score, fbeta_score, recall_score
import json

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
    
#     metricas = [('auc',roc_auc_score),
#             ('recall',recall_score),
#             ('f1_score',f1_score),
#             ('f2_score',fbeta_score)
#            ]


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
    RECALL = []
    f1 = []
    f2 = []
    modelo = []
    tp = []
    tn = []
    fp = []
    fn = []

    for m in hipers:
        print(m)
        clf = hipers[m].fit(X_train, y_train) #ajuste
        y_pred = clf.predict(X_test) #predicao
        
        cm = confusion_matrix(y_test, y_pred)
        modelo.append(m)
        
        if y_test.sum() == 0:
            AUC.append(np.nan)
            RECALL.append(np.nan)
            f1.append(np.nan)
            f2.append(np.nan)
            tp.append(np.nan)
            tn.append(np.nan)
            fp.append(np.nan)
            fn.append(np.nan)
        else:
            AUC.append(roc_auc_score(y_test, y_pred, average='weighted'))
            RECALL.append(recall_score(y_test, y_pred))
            f1.append(f1_score(y_test, y_pred))
            f2.append(fbeta_score(y_test, y_pred, beta=2))
            tp.append(cm[1,1])
            tn.append(cm[0,0])
            fp.append(cm[0,1])
            fn.append(cm[1,0])

      
    df = pd.DataFrame({'AUC':AUC,
                        'RECALL':RECALL,
                        'F1':f1,
                        'F2':f2,
                        'TN':tn,
                        'FN':fn,
                        'FP':fp,
                        'TP':tp,
                        'MODEL':modelo
                            })
    df['N_NEG'] = sum(y_test == 0)
    df['N_POS'] = sum(y_test == 1)
 
    return df