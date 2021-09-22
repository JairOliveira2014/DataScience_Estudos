
# Machine Learning - Classificação

# Definição do diretório de trabalho
setwd("~/Documents/Estudos/DSA/BigDataRAzure/Cap-11/Classificacao")
getwd()

# Definição do Problema de Negócio: Previsão de Ocorrência de Câncer de Mama
# Link para acesso em http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29

# Step 1 - Importando os dados

# Os dados do câncer da mama incluem 569 observações de biópsias de câncer, 
# cada um com 32 características (variáveis). Uma característica é um número de 
# identificação (ID), outro é o diagnóstico de câncer, e 30 são medidas laboratoriais 
# numéricas. O diagnóstico é codificado como "M" para indicar maligno ou "B" para 
# indicar benigno.

Xtrain <- read.csv("dataset.csv", stringsAsFactors = FALSE)
View(Xtrain)
str(Xtrain)

# Step 2 - Explorando os dados

# Excluindo a coluna ID
# Independentemente do método de aprendizagem de máquina, deve sempre ser excluídas 
# variáveis de ID. Caso contrário, isso pode levar a resultados errados porque o ID 
# pode ser usado para unicamente "prever" cada exemplo. Por conseguinte, um modelo 
# que inclui um identificador pode sofrer de superajuste (overfitting), 
# e será muito difícil usá-lo para generalizar outros dados.

Xtrain$id = NULL

# Ajustando o Label da Variável alvo
Xtrain$diagnosis <- sapply(Xtrain$diagnosis, function(x){ ifelse(x=="M", "Maligno", "Benigno")})

# Alterando o tipo de dado para Factor
table(Xtrain$diagnosis)
Xtrain$diagnosis <- factor(Xtrain$diagnosis, levels = c("Benigno", "Maligno"), labels = c("Benigno", "Maligno"))
str(Xtrain$diagnosis)

# Verificando a proporção
round(prop.table(table(Xtrain$diagnosis)) * 100, digits = 1)

# Medidas de Tendência Central
# Detectamos um problema de escala entre os dados, que então precisam ser normalizados
# O cálculo de distância feito pelo kNN é dependente das medidas de escala nos dados de entrada.
summary(Xtrain[c("radius_mean", "area_mean", "smoothness_mean")])

# Criando a função de normalização
norma <- function(x){ 
  return((x - min(x)) / (max(x) - min(x)))}

Xtrain_norm <- as.data.frame(lapply(Xtrain[2:31], norma))
View(Xtrain_norm)

# Step 3 - Treinando o Modelo KNN

# Carregando o pacote class
library(class)

# Realizando o split dos dados para treino e teste
train <- Xtrain_norm[1:469,]
test <- Xtrain_norm[470:569,]

# Criando as labels
train_labels <- Xtrain[1:469, 1]
test_labels <- Xtrain[470:569, 1]

length(train_labels)
length(test_labels)

# Criando o Modelo
model_knn_v1 <- knn(train = train, test = test, cl = train_labels, k = 21)

summary(model_knn_v1)

# Step 4 -  Avaliando e interpretando o modelo
#install.packages("gmodels")
library(gmodels)

# Criando uma tabela cruzada dos dados previstos x atuais
# Usaremos amostras com 100 observações
CrossTable(x = test_labels, y = model_knn_v1, prop.chisq = FALSE)

# Nosso Modelo Obteve uma ótima taxa de acerto

# Step 5 - Otimizando a performance do modelo

# Usando a função scale() para padronizar o z-score 
Xtrain_z <- as.data.frame(scale(Xtrain[-1]))

# Confirmando a Operação
summary(Xtrain_z$area_mean)

# Realizando o split dos dados para treino e teste
train_z <- Xtrain_z[1:469,]
test_z <- Xtrain_z[470:569,]

train_labels <- Xtrain[1:469, 1]
test_labels <- Xtrain[470:569, 1]

# Criando o Modelo
model_knn_v2 <- knn(train = train_z, test = test_z, cl = train_labels, k = 21)

summary(model_knn_v2)

# Criando uma tabela cruzada dos dados previstos x atuais
# Usaremos amostras com 100 observações
CrossTable(x = test_labels, y = model_knn_v2, prop.chisq = FALSE)

# Podemos observar que houve uma redução na accuracy

# Step 6 - Construindo o modelo de SVM

# Definindo a semente para resultados reproduzíveis
set.seed(40) 

# Prepara o dataset
dados <- read.csv("dataset.csv", stringsAsFactors = FALSE)
dados$id = NULL
dados[,'index'] <- ifelse(runif(nrow(dados)) < 0.8,1,0)
View(dados)

# Dados de treino e teste
trainset <- dados[dados$index==1,]
testset <- dados[dados$index==0,]

# Obter o índice 
trainColNum <- grep('index', names(trainset))

# Remover o índice dos datasets
trainset <- trainset[,-trainColNum]
testset <- testset[,-trainColNum]

# Obter índice de coluna da variável target no conjunto de dados
typeColNum <- grep('diag',names(dados))

# Cria o modelo
# Nós ajustamos o kernel para radial, já que este conjunto de dados não tem um 
# plano linear que pode ser desenhado
library(e1071)
?svm
modelo_svm_v1 <- svm(diagnosis ~ ., 
                     data = trainset, 
                     type = 'C-classification', 
                     kernel = 'radial') 


# Previsões

# Previsões nos dados de treino
pred_train <- predict(modelo_svm_v1, trainset) 

# Percentual de previsões corretas com dataset de treino
mean(pred_train == trainset$diagnosis)  


# Previsões nos dados de teste
pred_test <- predict(modelo_svm_v1, testset) 

# Percentual de previsões corretas com dataset de teste
mean(pred_test == testset$diagnosis)  

# Confusion Matrix
table(pred_test, testset$diagnosis)

# Podemos observar que nossa accuracy foi extremamente próxima ao KNN

# Estep 7: Construindo um Modelo com Algoritmo Random Forest

# Criando o modelo
library(rpart)
modelo_rf_v1 = rpart(diagnosis ~ ., data = trainset, control = rpart.control(cp = .0005)) 

# Previsões nos dados de teste
tree_pred = predict(modelo_rf_v1, testset, type='class')

# Percentual de previsões corretas com dataset de teste
mean(tree_pred==testset$diagnosis) 

# Confusion Matrix
table(tree_pred, testset$diagnosis)

# Nosso Algoritmo obteve um resultado mais baixo que nos outros modelos

# Testar vários algoritmos diferentes ajudar a perceber os dados por diferentes métodos
# Existem muitas formas de exploração, transformações e muitos outros métodos de implementação de modelos
# Por isso, acredito que experimentar é uma das melhores formas de conhecimento