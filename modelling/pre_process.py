import pandas as pd
import numpy as np

#Reamostragem
from imblearn.over_sampling import SMOTE, SMOTENC

#Preprocessamento de dados
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


# Pré processamento de dados

def pre_processing(df, apply_std=True, apply_oversample=False, p=.15):
    
    dados = df.copy()

    dados.drop(['isFlaggedFraud','isUnauthorizedOverdraft'], axis=1, inplace=True)

    # Listas de variaveis numericas e categoricas
    X_num = ['step', 'amount','oldBalanceOrig','newBalanceOrig', 'oldBalanceDest', 'newBalanceDest']
    X_cat = ['action', 'nameOrig','nameDest']
    X = dados.columns
    
    features = ['step', 'action', 'amount', 'nameOrig', 'oldBalanceOrig',
                'newBalanceOrig', 'nameDest', 'oldBalanceDest', 'newBalanceDest']

    # Dados categoricos:
    for i in X_cat:
        le = LabelEncoder()
        le.fit(dados[i])
        dados[i] = le.transform(dados[i])

    if apply_std:
        # Padronização APENAS das variaveis numericas
        treino_std = StandardScaler()
        numerical_variables_sdt = treino_std.fit(dados[X_num])
        numerical_variables_sdt = treino_std.transform(dados[X_num])
        dados[X_num] = numerical_variables_sdt
    
    # Dados de treino e teste
    treino, teste = train_test_split(dados, 
                                     test_size=.2, 
                                     stratify=dados['isFraud'], 
                                     random_state = 123)

    if apply_oversample:
        # Resmostragem de dados de treino (balancemanto de dados):
        smt = SMOTENC(sampling_strategy=p, 
                      random_state=123, 
                      categorical_features=[1,3,6], 
                      n_jobs = -1)
        
        X_oversampled, y_oversampled = smt.fit_resample(treino[features], treino['isFraud'])
        
        return X_oversampled, y_oversampled, teste
        
    return treino[features], treino['isFraud'], teste
 