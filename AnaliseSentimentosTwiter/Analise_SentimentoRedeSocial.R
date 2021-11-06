
# Análise de Sentimento - Naive Bayes

# Neste pequeno projeto iremos coletar dados da rede social Twitter, e assim realizar a análise de sentimento utilizando o algoritmo Naive Bayes. 
# Este é um tema à qual tem crescido de forma acelerada, pois, muitas organizações tem requisitado por tais análises em relação à produtos, 
# campanhas de marketing e até o sentimento quanto a compainha.

# Definindo diretório do trabalho
setwd("~/Documents/Estudos/DSA/BigDataRAzure/Cap-17")
getwd()

## Etapa 1 - Pacotes, acesso e autenticação

# Instalando e carregando os pacotes que iremos utilizar inicialmente
install.packages("twitteR")
install.packages("httr")
library(twitteR)
library(httr)

# Definindo as chaves de acesso ao Twitter
# Obs. Deve-se criar uma conta desenvolvedor e solicitar as chaves de acesso a API, as minhas chaves estou guardando em um script auxiliar chamado utils_2.
# Caso possua suas chaves, substitua diretamente nos valores das variáveis

source("utils_2.R")

key <- key
keysecret <- keysecret
token <- token
tokensecret <- tokensecret

# Autenticação. Responda 1 quando perguntado sobre utilizar direct connection.
setup_twitter_oauth(key, keysecret, token, tokensecret)
#?setup_twitter_oauth

## Etapa 2 - Conexão e coleta dos dados

# Como criei essa conta apenas para essa o projeto não tenho twitters na time line, por isso, vamos dar uma olhada na time line da data science academy, 
# onde estou a realizar esse curso! assim conseguimos verificar nossa conexão. lol

userTimeline("dsacademybr")

# Capturando os tweets
# Pode-se utilizar o tema de interesse, desde pesquisas de marcas, assuntos e etc. Neste caso faremos uma análise de sentimento em relação ao tema Big Data
# Para esse efeito utilizaremos 1500 twitters.
tema <- "BigData"
qtd_tweets <- 1500
lingua <- "en"
tweetdata = searchTwitter(tema, n = qtd_tweets, lang = lingua)

# Visualizando as primeiras linhas do objeto tweetdata
head(tweetdata)

## Etapa 3 - Tratamento dos dados coletados através de text mining

# Instalando o pacote para Text Mining.
install.packages("tm")
install.packages("SnowballC")
library(SnowballC)
library(tm)
options(warn=-1)

# Obtendo o texto
tweetlist = sapply(tweetdata, function(x) x$getText())

# Removendo caracteres especiais
tweetlist = gsub("(RT|via)((?:\\b\\W*@\\w+)+)", "", tweetlist)
# Removendo @
tweetlist = gsub("@\\w+", "", tweetlist)
# Removendo pontuação
tweetlist = gsub("[[:punct:]]", "", tweetlist)
# Removendo digitos
tweetlist = gsub("[[:digit:]]", "", tweetlist)
# Removendo links html
tweetlist = gsub("http\\w+", "", tweetlist)
# Removendo espacos desnecessários
tweetlist = gsub("[ \t]{2,}", "", tweetlist)
tweetlist = gsub("^\\s+|\\s+$", "", tweetlist)

# Covertendo para o objeto Corpus e realizando algumas transformações
tweetcorpus <- Corpus(VectorSource(tweetlist))
tweetcorpus <- tm_map(tweetcorpus, removePunctuation)
tweetcorpus <- tm_map(tweetcorpus, content_transformer(tolower))

# Removendo as stopWords
tweetcorpus <- tm_map(tweetcorpus, function(x)removeWords(x, stopwords()))

## Etapa 4 - Wordcloud, associação entre as palavras e dendograma

install.packages("RColorBrewer")
install.packages("wordcloud")
library(RColorBrewer)
library(wordcloud)

# Gerando uma nuvem palavras
pal <- brewer.pal(8,"Dark2")

wordcloud(tweetcorpus, 
          min.freq = 2, 
          scale = c(3,1), 
          random.color = F, 
          max.word = 80, 
          random.order = F,
          colors = pal)

# Convertendo o objeto texto para o formato de matriz
tweettdm <- TermDocumentMatrix(tweetcorpus)
tweettdm

# Encontrando as palavras que aparecem com mais frequência
findFreqTerms(tweettdm, lowfreq = 11)

# Buscando associações com o spark
findAssocs(tweettdm, 'python', 0.60)

# Removendo termos esparsos (não utilizados frequentemente)
tweet2tdm <- removeSparseTerms(tweettdm, sparse = 0.9)

# Criando escala nos dados
tweet2tdmscale <- scale(tweet2tdm)

# Distance Matrix
tweetdist <- dist(tweet2tdmscale, method = "euclidean")

# Preparando o dendograma
tweetfit <- hclust(tweetdist)

# Criando o dendograma (verificando como as palavras se agrupam)
plot(tweetfit)

# Verificando os grupos
cutree(tweetfit, k = 4)

# Visualizando os grupos de palavras no dendograma
rect.hclust(tweetfit, k = 3, border = "red")

## Etapa 5 - Usando Classificador Naive Bayes para análise de sentimento

# https://cran.r-project.org/src/contrib/Archive/Rstem/
# https://cran.r-project.org/src/contrib/Archive/sentiment/

install.packages("Mini-Projeto01/Rstem_0.4-1.tar.gz", sep = "", repos = NULL, type = "source")
install.packages("Mini-Projeto01/sentiment_0.2.tar.gz",sep = "", repos = NULL, type = "source")
install.packages("ggplot2")
library(Rstem)
library(sentiment)
library(ggplot2)

# Antes de realizar a classificação teremos que colocar twitterlist em lower case

# Criando função para tolower
try.error = function(x)
{
  # Criando missing value
  y = NA
  try_error = tryCatch(tolower(x), error=function(e) e)
  if (!inherits(try_error, "error"))
    y = tolower(x)
  return(y)
}

# Lower case
tweetlist_l = sapply(tweetlist, try.error)

# Removendo os NAs
tweetlist_l = tweetlist_l[!is.na(tweetlist_l)]
names(tweetlist_l) = NULL

# Classificando emocao
class_emo = classify_emotion(tweetlist_l, algorithm = "bayes", prior = 1.0)
emotion = class_emo[,7]

# Substituindo NA's por "Neutro"
emotion[is.na(emotion)] = "Neutro"

# Classificando polaridade
class_pol = classify_polarity(tweetlist, algorithm = "bayes")
polarity = class_pol[,4]

# Gerando um dataframe com o resultado
sent_df = data.frame(text = tweetlist_l, emotion = emotion,
                     polarity = polarity, stringsAsFactors = FALSE)

# Ordenando o dataframe
sent_df = within(sent_df,
                 emotion <- factor(emotion, levels = names(sort(table(emotion), 
                                                                decreasing=TRUE))))


# Emoções encontradas
ggplot(sent_df, aes(x = emotion)) +
  geom_bar(aes(y = ..count.., fill = emotion)) +
  scale_fill_brewer(palette = "Dark2") +
  labs(x = "Categorias", y = "Numero de Tweets") 

# Polaridade
ggplot(sent_df, aes(x=polarity)) +
  geom_bar(aes(y=..count.., fill=polarity)) +
  scale_fill_brewer(palette="RdGy") +
  labs(x = "Categorias de Sentimento", y = "Numero de Tweets")


# Como podemos observar, esse script apenas possui de forma simples o entendimento de um processo de análise de sentimento, há inúmeras técnicas de feature engineering para tratamento 
# e vetorização de dados em textos à qual pode-se de deve-se realizar em um projeto mais robusto. No entanto, para contato inicial considero que o objetivo foi alcançado.

