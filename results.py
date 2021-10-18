#Carregando bibliotecas
import pandas as pd
import numpy as np

from modelling.pre_process import pre_processing
from modelling.fit import fit_models
from modelling.run import run_through_samples

from zipfile import ZipFile

import warnings
warnings.filterwarnings("ignore")

#Carregando dados
zip_file = ZipFile('cenario4.zip')
dados = {text_file.filename: pd.read_csv(zip_file.open(text_file.filename))
       for text_file in zip_file.infolist()
       if text_file.filename.endswith('.csv')}

f = run_through_samples(dados, over_sampling=False, prop=.1)
f.to_csv('Resultados/resultados_cenario4.csv', index=False)

