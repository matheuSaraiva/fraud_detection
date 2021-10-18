import pandas as pd
import numpy as np

#timing
from datetime import datetime

from modelling.pre_process import pre_processing
from modelling.fit import fit_models

def run_through_samples(cenario, over_sampling=True, prop=.2):
    
  
    results = []
    
    features = ['step', 'action', 'amount', 'nameOrig', 'oldBalanceOrig',
                'newBalanceOrig', 'nameDest', 'oldBalanceDest', 'newBalanceDest']
    
    for sample in np.sort(list(cenario)):
        t0 = datetime.now()
        f = cenario[sample]['isFraud']==1
        
        # Pular amostras que não contém as duas classes
        if f.sum() == 0:
            continue
        
        #dados de treino e de teste
        x_treino, y_treino, teste = pre_processing(df=cenario[sample], 
                                                apply_std=True, 
                                                apply_oversample=over_sampling, 
                                                p=prop)
        
        
        #ajuste, predicao e resultados
        this_result = fit_models(X_train=x_treino, 
                               y_train=y_treino, 
                               X_test=teste[features], 
                               y_test=teste['isFraud'])
        
        
        t1 = datetime.now()
        print(f'{sample[:-4]}: \n tempo de modelagem: {t1-t0}')
        
        
        this_result[['cenario/amostra']] = sample[:-4]
        
        results.append(this_result)
        pd.concat(results, ignore_index=True).to_csv('Resultados/resultados_preliminares_cenario4.csv', index=False)

        
    final = pd.concat(results, ignore_index=True)
    
        
    return final