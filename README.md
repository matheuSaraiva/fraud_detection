# Repósitório para o desenvolvimento do código referente a comparação de modelos de classificação para detecção de fraudes financeiras

Neste repositório se encontram todos os códigos, banco de dados e análises realizadas referentes ao desenvolvimento da metodologia e dos resultados de minha dissertação de mestrado.

## Estrutura

A execução deste projeto é dada através do arquivo *results.py*, que por sua vez carrega e executa os seguintes scrips em python, contidos no diretório "/modelling/":
- **pre_process.py**: Rotina criada para o tratamento de dados amostrais e separação dos mesmos em cojuntos de treino e de teste.
- **fit_models.py**: Rotina criada para ajustar modelos e gerar os resultados referentes aos seus desempenhos em predição, sob as métricas de AUC, Acurácia Balanceada, F1-score e F2-score. 
- **run.py**: Este script carrega e executa os scripts anteriores em um conjunto de amostras. Parte deste script carrega parâmetros de modelos já hiperparametrizados, que se encontram no diretório "/Hiperparametrizados/".

O scrip *results.py* é o script pricipal, responsável pela integração dos demais scripts e parametrização. Durante sua execução uma tabela em formato .csv é gerada, permintindo análises dos resultados parciais enquanto todo o conjunto de amostras ainda não foi finalizado. Ao fim de sua execução, uma tabela final em formato .csv é gerada com todos os resultados de todos os modelos em todas as amostras; armazeanada no diretório "/Resultados/".

## Dados

Os dados utilizados neste repositório são dados artificiais gerados pelo simulador AMLSim, [disponível aqui](https://github.com/IBM/AMLSim). A partir de diferentes parametrizações 3 cenários foram gerados, cada um com 1.000 amostras:

| Cenário | Horizonte Temporal | multiplicador | N. Clientes | N. Fraudadores | N. Mercados | N. Bancos | Prob. Fraude | Limite Transf. |
|---------|--------------------|---------------|-------------|----------------|-------------|-----------|--------------|----------------|
| 1       | 1000               | 0.5           | 50          | 50             | 500         | 5         | 0.01         | 50000          |
| 2       | 10000              | 1             | 15          | 100            | 100         | 2         | 0.001        | 2.000.00.000   |

No script de execução *results.py* cada cenário é inserido de maneira compactada em formato .zip.

## Modelos

os modelos utilizados neste projeto são:
- Regressão Logística
- Random Forests
- Redes Neurais Artificiais (MLP)
- Support Vector Machine (SVM)
- Extreme Gradiente Boosting Machine (XGB)
- Sistemas Baseados em Regras Fuzzy (SBRF)

Os ajustes, predições e avaliações de desempenho de todos os modelos, com excessão do modelo SBRF, são feitos através da biblioteca [Scikit-learn](https://scikit-learn.org/stable/). O modelo SBRF é ajustado e executado sob a liguagem de programação R, através da utilzação dos pacotes [frbs](https://cran.r-project.org/web/packages/frbs/frbs.pdf) e  [Tidyverse](https://www.tidyverse.org/).

## Análises

No diretório "/Resultados/" se encontram os arquivos gerados pelo script "results.py" e o jupyter notebook "analise_inicial.ipynb" que contém uma análise de uma das amostras e também do desempenho dos modelos
