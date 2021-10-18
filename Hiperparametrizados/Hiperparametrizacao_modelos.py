#!/usr/bin/env python
# coding: utf-8

# # Bibliotecas

# In[ ]:


import pandas as pd

#Reamostragem
from imblearn.over_sampling import SMOTE, SMOTENC

#Lista de modelos
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
# from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

#Preprocessamento de dados
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#hiperparametrizacao
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV #hiperparametrizacao

#Salvar e carregar modelos treinados
from joblib import dump, load

#Comparacao de modelos
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, auc

#timing
from datetime import datetime

#graficos
from zipfile import ZipFile
import seaborn as sns


# In[ ]:


zip_file = ZipFile('cenario1.zip')


# In[ ]:


dados = {text_file.filename: pd.read_csv(zip_file.open(text_file.filename))
       for text_file in zip_file.infolist()
       if text_file.filename == 'cenario1/resultado_1.csv'}

for i in dados:
    dados[i].drop(['isFlaggedFraud','isUnauthorizedOverdraft'], axis=1, inplace=True)
    
dados = dados['cenario1/resultado_1.csv']


# In[ ]:


dados['isFraud'].value_counts()


# # Pré processamento de dados

# In[ ]:


def pre_processing(df, apply_std=True, apply_oversample=True, p=.15):
    
    dados = df.copy()
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
        smt = SMOTENC(sampling_strategy=p, random_state=123, categorical_features=[1,3,6], n_jobs = -1) 
        X_oversampled, y_oversampled = smt.fit_resample(treino[features], treino['isFraud'])
        
        return X_oversampled, y_oversampled, teste
        
    return treino[features], treino['isFraud'], teste
 


# # Hiperparametrização de modelos

# In[ ]:


models = [
          LogisticRegression(), 
          RandomForestClassifier(),
          XGBClassifier(),
          MLPClassifier(),
          SVC()
         ]


# In[ ]:


params_logit = {'penalty':['l1', 'l2', 'elasticnet'],
                'C':[.5, 1, 3, 10],
                'class_weight':['balanced', None]                 
               }

params_svc = {'C':[.5, 1, 3]
             }

params_mlp = {'hidden_layer_sizes':[(10, ), (50,), (100, ), (150, ), (200, )],
              'activation':['logistic', 'tanh', 'relu']
              }

params_xgb = {'n_estimators':[10, 50, 100, 150, 200, 400],
              'booster':['gbtree', 'gblinear', 'dart'],
              'importance_type':["gain", "weight", "cover", "total_gain","total_cover"]
              }

params_rf = {'n_estimators':[10, 50, 100, 150, 200, 400],
             'criterion':["gini", "entropy"],
             'class_weight':["balanced", "balanced_subsample"]
            }

h_params = [
            params_logit, 
            params_rf,
            params_xgb, 
            params_mlp, 
            params_svc
           ]


# In[ ]:


setups = []
for m in zip(models, h_params):
    model = GridSearchCV(estimator=m[0], param_grid=m[1], n_jobs=-1, scoring='roc_auc', verbose=2)
    setups.append(model)


# # Estimação de parametros

# In[ ]:


x_treino, y_treino, teste = pre_processing(df=dados, 
                                                apply_std=True, 
                                                apply_oversample=True, 
                                                p=.15)


# In[ ]:


clf = []
for i in setups:
    t0 = datetime.now()
    m = i.fit(X=x_treino, y=y_treino)
    clf.append(m)
    dump(m, f'{str(i.estimator)[:4]}.joblib')
    print(f'Tempo de modelagem: {datetime.now() - t0}')

dump(clf, 'All_models.joblib')





