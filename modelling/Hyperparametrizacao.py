#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

#Lista de modelos
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
# from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

#Preprocessamento de dados
from pre_process import pre_processing

#hiperparametrizacao
from sklearn.model_selection import GridSearchCV #hiperparametrizacao

#Salvar e carregar modelos treinados
from joblib import dump, load

#Comparacao de modelos
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer

#timing
from datetime import datetime


# In[3]:


dados = pd.read_csv('amostra_39.csv')
dados['isFraud'].value_counts()


# In[4]:


# # Pré processamento de dados
# # Hiperparametrização de modelos
models = [
          LogisticRegression(), 
          RandomForestClassifier(),
          XGBClassifier(),
          MLPClassifier(),
          SVC()
         ]

params_logit = {'penalty':['l1', 'l2', 'elasticnet'],
                'C':[.5, 1, 3, 10],
                'class_weight':['balanced', None]                 
               }

params_svc = {'C':[.5, 1, 3], 
              'kernel':['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
              
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

setups = []

f2 = make_scorer(fbeta_score , beta=2)


# In[5]:


for m in zip(models, h_params):
    model = GridSearchCV(estimator=m[0], param_grid=m[1], n_jobs=-1, scoring=f2, verbose=2)
    setups.append(model)


# In[ ]:


x_treino, y_treino, teste = pre_processing(df=dados, 
                                                apply_std=True, 
                                                apply_oversample=True, 
                                                p=.15)

clf = []
for i in setups:
    t0 = datetime.now()
    m = i.fit(X=x_treino, y=y_treino)
    clf.append(m)
    dump(m, f'{str(i.estimator)[:4]}.joblib')
    print(f'Tempo de modelagem: {datetime.now() - t0}')

dump(clf, 'All_models.joblib')

