
# Análise de de concessão de crédito

# Neste estudo iremos analisar um conjunto de dados de crédito status e criar um modelo preditivo para classificar a concesão de crédito aos indivíduos.

# Configurando o diretório de trabalho
setwd("~/Documents/Estudos/DSA/BigDataRAzure/Cap-18/5-Mini-Projeto4")
getwd()

# Etapa 1 - Carregamento dos dados e percepção dos mesmos

# Carregando o DataSet 
df <- read.csv("credit_dataset.csv")

# Verificando nossos dados
head(df)
str(df)
View(df)

# Etapa 2 - Análise exploratória e visualização

# Analizando a distribuição dos dados

summary(df)

# como podemos observar muitas das features são categóricas, porém, foram atribuídas como "int" pelo R. Outro pronto que devemos tratar será as variáveis "age" e "credit.amount",
# neste caso tratar os dados e criarmos intervalos será uma boa opção para o tipo de problemática ou normaliza-los.

# Analizando a proporção dos dados pelo rating
table(df$credit.rating)

# Como podemos observar há um desequilibrio entre os dados.

table(is.na(df))

# Como podemos observar em nosso conjunto de dados não há valores missing.

# Análise de Correlação

# Definindo as colunas para a análise de correlação 
cols <- names(df)

# Métodos de Correlação
# Pearson - coeficiente usado para medir o grau de relacionamento entre duas variáveis com relação linear
# Spearman - teste não paramétrico, para medir o grau de relacionamento entre duas variaveis
# Kendall - teste não paramétrico, para medir a força de dependência entre duas variaveis

# Vetor com os métodos de correlação
metodos <- c("pearson", "spearman")

# Aplicando os métodos de correlação com a função cor()
cors <- lapply(metodos, function(method) 
  (cor(df[, cols], method = method)))

head(cors)

# Preprando o plot
require(lattice)
plot.cors <- function(x, labs){
  diag(x) <- 0.0 
  plot( levelplot(x, 
                  main = paste("Plot de Correlação usando Método", labs),
                  scales = list(x = list(rot = 90), cex = 1.0)) )
}

# Mapa de Correlação
Map(plot.cors, cors, metodos)

# Podemos observar que há algumas correlações negativas e poucas positivas, mesmo analisando os dois métodos.

# Verificando alguns plots em relação a variável target, no entanto, teremos que transformar em factores algumas variáveis
categorical.vars <- c('credit.rating', 'account.balance', 'previous.credit.payment.status',
                      'credit.purpose', 'savings', 'employment.duration', 'installment.rate',
                      'marital.status', 'guarantor', 'residence.duration', 'current.assets',
                      'other.credits', 'apartment.type', 'bank.credits', 'occupation', 
                      'dependents', 'telephone', 'foreign.worker')

## Convertendo as variáveis para o tipo fator (categórica)
to.factors <- function(df, variables){
  for (variable in variables){
    df[[variable]] <- as.factor(df[[variable]])
  }
  return(df)
}

df.fact <- to.factors(df = df, variables = categorical.vars)
str(df.fact)

# Carregando e visualizando os dados
library(ggplot2)

lapply(cols, function(x){
  if(is.factor(df.fact[,x])) {
    ggplot(df.fact, aes_string(x)) +
      geom_bar() + 
      facet_grid(. ~ credit.rating) + 
      ggtitle(paste("Total de concessão de Crédito Sim/Não por",x))}})

# Plots credit.ratings vs outras variáveis
lapply(cols, function(x){
  if(is.factor(df.fact[,x]) & x != "credit.rating") {
    ggplot(df.fact, aes(credit.rating)) +
      geom_bar() + 
      facet_grid(paste(x, " ~ credit.rating"))+ 
      ggtitle(paste("Total de concessão de Crédito sim/não credit.rating e",x))
  }})

# Etapa 3 - Transformação dos dados

## Para início vamos começar normalizando algumas variáveis

to.norm <- function(df, variables) {
  for (variable in variables) {
    df[[variable]] <- scale(df[[variable]], center = T, scale = T)
  }
  return(df)
}
var.norm <-  c("credit.duration.months", "age", "credit.amount")
df.norm <- to.norm(df = df, variables = var.norm)   

View(df.norm)

# Agora que já temos um conjunto de dados "ajustado" podemos agora realizar feature selection para construir um modelo e realizar algumas previsões para servir de base line

# Dividindo em dados de treino e teste
indexes <- sample(1:nrow(df.norm), size = 0.6 * nrow(df.norm))
treino <- df.norm[indexes,]
teste <- df.norm[-indexes,]

# agora utilizaremos o random forest para realizar o feature selection
library(caret)
# Avalidando a importância de todas as variaveis
modelo <- randomForest(credit.rating ~ . , 
                       data = treino, 
                       ntree = 500, 
                       nodesize = 10,
                       importance = TRUE)

# Plotando as variáveis por grau de importância
varImpPlot(modelo)

# Agora conseguimos visualizar algumas features consideradas importantes para nosso modelo e vamos assim realizar um seleção e criar o nosso modelo, 
# sinta-se a vontade para criar sua seleção de features
feature.import <- c("account.balance", "credit.duration.months", "credit.amount", "credit.assets", "credit.purpose", "installment.rate",
                    "employment.duration", "previous.credit.payment.status", "savins", "occupation")


# Etapa 4 - Criação do modelo em random forest

model <- randomForest(credit.rating ~ account.balance + credit.duration.months + credit.amount + current.assets + credit.purpose +
                        installment.rate + employment.duration + previous.credit.payment.status + savings + occupation,
                      data = treino,
                      ntree = 1000,
                      nodesize = 10,
                      proximity = TRUE)
print(model)

# realizando as predições
result <- predict(model, newdata=teste, type = "response")
result <- round(result)

# Avaliando o modelo
confusionMatrix(table(result, teste$credit.rating), positive = "1")

# Como podemos observar nosso modelo obteve uma acuracy em torno de 0.75, isso já é o suficiente para sabermos que há possibilidades em avançar e melhorar o modelo,
# Seja em técnicas de feature engineering, transformações, novos algoritmos e ajuste dos hiperparâmentros.
# No entanto, neste momento ainda não avançarei a estes tópicos e continuarei a desenvolver minhas habilidades em R e alguns conceitos.