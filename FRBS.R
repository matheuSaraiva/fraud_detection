########################## BASE DE REGRAS VIA AI ######################
#setwd('/home/matheus/Dropbox/UFLA/materiais dissertacao/Modelagem/')

# CARREGANDO PACOTES ------------------------------------------------------
library(frbs); library(tidyverse); library(caret); library(ROCR); library(pROC)

# FUNCOES
load_df = function(n_df){
  # Function to load datasets already processed
  X_train = paste0('treinos_testes/x_treino_resultado_',n_df,'.csv')
  y_train = paste0('treinos_testes/y_treino_resultado_',n_df,'.csv')
  test = paste0('treinos_testes/teste_resultado_',n_df,'.csv')
  
  X_train = read.csv(X_train)
  y_train = read.csv(y_train)
  test = read.csv(test)
  
  y_train$isFraud = ifelse(y_train$isFraud == 1, 2, 1)
  test$isFraud = ifelse(test$isFraud == 1, 2, 1)
  return(list(X_train, y_train, test))
}
run_fuzzy_model = function(X_train, y_train){
  
  y_train$isFraud = ifelse(y_train$isFraud == 1, 2, 1)
  amplitudes = apply(X_train, 2, range)
  
  # CONTROLADOR FUZZY  ------------------------------------------------------
  controlador = list(num.labels=3,
                     type.mf="SIGMOID",
                     type.tnorm="PRODUCT",
                     type.implication.func="ZADEH",
                     type.defuz='COG',
                     num.class=3,
                     name='Fraudes')
  
  # MODELAGEM ---------------------------------------------------------------
  now = Sys.time()
  fuzzy_classificacao <- frbs.learn(data.train=cbind(X_train, y_train), 
                                    range.data=amplitudes,
                                    method.type="FRBCS.CHI",
                                    control=controlador)
  
  print(cat('Tempo de ajuste: ', (Sys.time() - now) ))
  
  return(fuzzy_classificacao)
}
results = function(n){
  
  
  
  ctrl = trainControl(method = "cv",
                     number = 5, 
                     returnResamp = 'none',
                     summaryFunction = twoClassSummary,
                     classProbs = T,
                     savePredictions = T,
                     verboseIter = F)

    
    fuzzy_params = expand.grid(num.labels=3,
                            type.mf="GAUSSIAN",
                            type.tnorm="MIN",
                            type.implication.func="ZADEH",
                            type.defuz='COG',
                            num.class=3,
                            name='Fraudes') 
    
    set.seed(123)
    model <- train(isFraud ~ .,
                  data = cbind(cbind(X_train, y_train)),
                  method = "FRBCS.W", 
                  metric = "AUC",
                  tuneGrid = fuzzy_params,
                  verbose = FALSE,
                  trControl = ctrl)
    
  preds = predict(model, test[, -ncol(test)])
  trues = test$isFraud
  
  
  CM = confusionMatrix(as.factor(preds), as.factor(trues))
  balanced_accuracy = cm[[4]][11]
  AUC = auc(trues, preds)
}
bring_results = function(n_df){
  #TREINO E TESTE
  dfs = load_df(n_df)
  X_train = dfs[1] %>% data.frame()
  y_train = dfs[2] %>% data.frame()
  test = dfs[3] %>% data.frame()
  
  #AJUSTE DO MODELO
  modelo = run_fuzzy_model(X_train, y_train)
  
  # #Predicao
  t0 = Sys.time()
  preds = predict(modelo, test[,-ncol(test)])
  t1 = Sys.time()
  print(cat('Tempo predicao: ',(t1-t0)))
  trues = test$isFraud
  
  cm = table(preds, trues)
  TPR = cm[2,2] / (cm[2,2]+cm[1,2])
  TNR = cm[1,1] / (cm[1,1]+cm[2,1])
  
  balanced_accuracy = (TPR+TNR)/2
  AUC = auc(trues, as.numeric(preds))
  
  
  score = numeric()
  parameter = numeric()
  
  
  d = data.frame(as.numeric(balanced_accuracy), as.numeric(AUC))
  names(d) = c('balanced_accuracy', 'auc')
  
  return(d)
  
}

# TESTANDO ----------------------------------------------------------------
hiper = function(n_df, LABELS, MF, TNORM, IMPLICATION, 
                 DEFUZ, CLASS, METHOD){
  
  #TREINO E TESTE
  dfs = load_df(n_df)
  X_train = dfs[1] %>% data.frame()
  y_train = dfs[2] %>% data.frame()
  test = dfs[3] %>% data.frame()
  
  #AJUSTE DO MODELO
    y_train$isFraud = ifelse(y_train$isFraud == 1, 2, 1)
    amplitudes = apply(X_train, 2, range)
    
   
    
  scores = data.frame(matrix(ncol = 9, nrow = 0))
  names(scores) = c('LABEL','MF','TNORM','IMPLICATION','DEFUZ','CLASS','METHOD','SCORE','TIME')
    
  # MODELAGEM ---------------------------------------------------------------
    now = Sys.time()
    
    for (label in LABELS) {
      for (mf in MF) {
        #for (tn in TNORM) {
          for (imp in IMPLICATION) {
            for (def in DEFUZ) {
              #for (cl in CLASS) {
                #for (mt in METHOD) {
                  t0 = Sys.time()
                  print(cat('LABEL: ', label,' \n',
                            'MF: ',mf, '\n',
                            #'TNORM: ',tn, '\n',
                            'IMPLICATION: ',imp, '\n',
                            'DEFUZ: ',def, '\n',
                            #'CLASS: ',cl, '\n',
                            #'METHOD: ',mt, '\n',
                            ''))
                  # CONTROLADOR FUZZY  ------------------------------------------------------
                  controlador = list(num.labels=label,
                                     type.mf=mf,
                                     type.tnorm='MIN',
                                     type.implication.func=imp,
                                     type.defuz=def,
                                     num.class=2,
                                     name='Fraudes')
                  
                  
                  fuzzy_classificacao <- frbs.learn(data.train=cbind(X_train, y_train), 
                                                    range.data=amplitudes,
                                                    method.type='FRBCS.CHI',
                                                    control=controlador)
                  
                  
                  preds = predict(fuzzy_classificacao, test[,-ncol(test)])
                  trues = test$isFraud
                  
                  cm = table(preds, trues)
                  TPR = cm[2,2] / (cm[2,2]+cm[1,2])
                  TNR = cm[1,1] / (cm[1,1]+cm[2,1])
                  
                  balanced_accuracy = (TPR+TNR)/2
                  AUC = auc(trues, as.numeric(preds))
                  t1 = Sys.time() - t0
                  scores[nrow(scores) + 1,] = c(label, mf, 'MIN', imp, def, 2, 'FRBS.CHI', AUC, t1)
                  write.csv(scores, 'hiperfuzzy.csv', row.names = F)
                  
                #}
              #}
            #}
          }
        }
      }
    }
    return(scores)
}

param1 = c(6,7,8)
param2 = c('GAUSSIAN')
param3 = c('MIN')
param4 = c('GOGUEN', 'GODEL', 'SHARP')
param5 = c('WAM', 'FISRT.MAX', 'LAST.MAX', 'MEAN.MAX', 'COG')
param6 = c(3)
param7 = c('FRBCS.CHI')

fuzzy_results = hiper(n_df = 1,
                      LABELS = param1,
                      MF = param2,
                      TNORM = param3,
                      IMPLICATION = param4,
                      DEFUZ = param5,
                      CLASS = param6,
                      METHOD = param7)









