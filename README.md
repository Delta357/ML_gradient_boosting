# ML_gradient_boosting

[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)
[![author](https://img.shields.io/badge/author-RafaelGallo-red.svg)](https://github.com/RafaelGallo?tab=repositories) 
[![](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-374/) 
[![](https://img.shields.io/badge/R-3.6.0-red.svg)](https://www.r-project.org/)
[![](https://img.shields.io/badge/ggplot2-white.svg)](https://ggplot2.tidyverse.org/)
[![](https://img.shields.io/badge/dplyr-blue.svg)](https://dplyr.tidyverse.org/)
[![](https://img.shields.io/badge/readr-green.svg)](https://readr.tidyverse.org/)
[![](https://img.shields.io/badge/ggvis-black.svg)](https://ggvis.tidyverse.org/)
[![](https://img.shields.io/badge/Shiny-red.svg)](https://shiny.tidyverse.org/)
[![](https://img.shields.io/badge/plotly-green.svg)](https://plotly.com/)
[![](https://img.shields.io/badge/XGBoost-red.svg)](https://xgboost.readthedocs.io/en/stable/#)
[![](https://img.shields.io/badge/Tensorflow-orange.svg)](https://powerbi.microsoft.com/pt-br/)
[![](https://img.shields.io/badge/Keras-red.svg)](https://powerbi.microsoft.com/pt-br/)
[![](https://img.shields.io/badge/CUDA-gree.svg)](https://powerbi.microsoft.com/pt-br/)
[![](https://img.shields.io/badge/Caret-orange.svg)](https://caret.tidyverse.org/)
[![](https://img.shields.io/badge/Pandas-blue.svg)](https://pandas.pydata.org/) 
[![](https://img.shields.io/badge/Matplotlib-blue.svg)](https://matplotlib.org/)
[![](https://img.shields.io/badge/Seaborn-green.svg)](https://seaborn.pydata.org/)
[![](https://img.shields.io/badge/Matplotlib-orange.svg)](https://scikit-learn.org/stable/) 
[![](https://img.shields.io/badge/Scikit_Learn-green.svg)](https://scikit-learn.org/stable/)
[![](https://img.shields.io/badge/Numpy-white.svg)](https://numpy.org/)
[![](https://img.shields.io/badge/PowerBI-red.svg)](https://powerbi.microsoft.com/pt-br/)

![Logo](https://img.freepik.com/vetores-gratis/banner-digital-da-rede-de-nervos-rosa_53876-117495.jpg?w=740&t=st=1685900259~exp=1685900859~hmac=2683870f7a342b523d288e5980b33aa3969f286597a1f0867b880467f83f068d)

## Introdução
Este repositório é um estudo sobre os algoritmos de machine learning aplicados ao Gradient Boosting com foco em processamento de linguagem natural. O Gradient Boosting é uma técnica utilizada para criar modelos mais poderosos, combinando modelos mais simples, como KNN, Naive Bayes e Árvore de Decisão. Os métodos de boosting auxiliam na melhoria do desempenho desses modelos mais robustos.

## Metodologia
Para realizar este estudo, foram coletados conjuntos de dados relevantes em processamento de linguagem natural. Em seguida, foi realizada uma análise exploratória dos dados, a fim de compreender suas características e distribuições. Posteriormente, foi realizado o pré-processamento dos dados, que envolveu etapas como limpeza, tokenização, remoção de stopwords, normalização e codificação de recursos.

## Pré-processamento
O pré-processamento dos dados é uma etapa fundamental para obter resultados precisos e confiáveis nos modelos de Processamento de Linguagem Natural. Neste estudo, aplicamos diversas técnicas de pré-processamento, como remoção de caracteres especiais, tokenização, remoção de stop words e normalização de texto.
A remoção de caracteres especiais envolveu a eliminação de sinais de pontuação, caracteres especiais e outros elementos que não contribuem para a análise de texto. Em seguida, realizamos a tokenização, que consiste em dividir o texto em palavras individuais, para facilitar o processamento subsequente.
Além disso, removemos as stop words, que são palavras comuns e pouco informativas, como "é", "em" e "com". Essas palavras não agregam muito valor à análise de texto e, portanto, foram eliminadas.

## Modelo ML
O próximo passo foi a implementação dos algoritmos de Gradient Boosting, utilizando bibliotecas como Scikit-learn ou XGBoost. Os modelos foram treinados e avaliados com base em métricas de desempenho, como precisão, recall, F1-score e acurácia. Além disso, foram utilizadas técnicas de validação cruzada para avaliar a generalização dos modelos.

## Conclusão
Neste estudo, foi demonstrado o uso do Gradient Boosting como uma técnica eficaz para melhorar a performance de modelos em processamento de linguagem natural. Os algoritmos de boosting, combinados com modelos mais simples, mostraram-se capazes de produzir resultados mais precisos e confiáveis.
No entanto, é importante ressaltar que a escolha dos algoritmos de Gradient Boosting e dos modelos base pode variar de acordo com o contexto do problema e as características dos dados. Portanto, é recomendado realizar experimentos adicionais e explorar diferentes abordagens para obter os melhores resultados em aplicações específicas de processamento de linguagem natural.
Este estudo serve como ponto de partida e referência para futuras pesquisas e aplicações no campo de machine learning e processamento de linguagem natural, auxiliando na compreensão dos algoritmos de Gradient Boosting e em sua aplicação prática.

## Stack utilizada

**Programação** Python, R.

**Machine learning**: Scikit-learn.

**Leitura CSV**: Pandas.

**Análise de dados**: Seaborn, Matplotlib.

**Modelo machine learning - Processo de linguagem natural**: NLTK, TextBlob, Vander.

## Dataset

| Dataset               | Link                                                |
| ----------------- | ---------------------------------------------------------------- |
|  |[Projeto - Notebook]()|
|  |[Projeto - Notebook]()|
|  |[Projeto - Notebook]()|
|  |[Projeto - Notebook]()|
|  |[Projeto - Notebook]()|



## Variáveis de Ambiente

Para rodar esse projeto, você vai precisar adicionar as seguintes variáveis de ambiente no seu .env

`API_KEY`

`ANOTHER_API_KEY`


## Instalação

Instalação das bibliotecas para esse projeto no python.

```bash
  conda install pandas 
  conda install scikitlearn
  conda install numpy
  conda install scipy
  conda install matplotlib

  python==3.6.4
  numpy==1.13.3
  scipy==1.0.0
  matplotlib==2.1.2
```
Instalação do Python É altamente recomendável usar o anaconda para instalar o python. Clique aqui para ir para a página de download do Anaconda https://www.anaconda.com/download. Certifique-se de baixar a versão Python 3.6. Se você estiver em uma máquina Windows: Abra o executável após a conclusão do download e siga as instruções. 

Assim que a instalação for concluída, abra o prompt do Anaconda no menu iniciar. Isso abrirá um terminal com o python ativado. Se você estiver em uma máquina Linux: Abra um terminal e navegue até o diretório onde o Anaconda foi baixado. 
Altere a permissão para o arquivo baixado para que ele possa ser executado. Portanto, se o nome do arquivo baixado for Anaconda3-5.1.0-Linux-x86_64.sh, use o seguinte comando: chmod a x Anaconda3-5.1.0-Linux-x86_64.sh.

Agora execute o script de instalação usando.


Depois de instalar o python, crie um novo ambiente python com todos os requisitos usando o seguinte comando

```bash
conda env create -f environment.yml
```
Após a configuração do novo ambiente, ative-o usando (windows)
```bash
activate "Nome do projeto"
```
ou se você estiver em uma máquina Linux
```bash
source "Nome do projeto" 
```
Agora que temos nosso ambiente Python todo configurado, podemos começar a trabalhar nas atribuições. Para fazer isso, navegue até o diretório onde as atribuições foram instaladas e inicie o notebook jupyter a partir do terminal usando o comando
```bash
jupyter notebook
```

## Demo modelo gradient boosting

```
# Importação das bibliotecas de nlp
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Gerar dados de exemplo
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar o classificador Gradient Boosting
gb_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

# Treinar o classificador
gb_classifier.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = gb_classifier.predict(X_test)

# Calcular a precisão do modelo
accuracy = accuracy_score(y_test, y_pred)
print("Precisão do modelo:", accuracy)

## Aplicação em R
library(xgboost)

# Gerar dados de exemplo
data <- matrix(rnorm(1000), ncol = 10)
labels <- sample(c(0, 1), 100, replace = TRUE)

# Dividir os dados em conjunto de treinamento e teste
train_indices <- sample(1:100, 80)
train_data <- data[train_indices, ]
train_labels <- labels[train_indices]
test_data <- data[-train_indices, ]
test_labels <- labels[-train_indices]

# Criar a matriz de dados específica do xgboost
dtrain <- xgb.DMatrix(data = as.matrix(train_data), label = train_labels)
dtest <- xgb.DMatrix(data = as.matrix(test_data), label = test_labels)

# Definir os parâmetros do modelo
params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eta = 0.1,
  max_depth = 3,
  nthread = 2,
  eval_metric = "error"
)

# Treinar o modelo
model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100
)

# Fazer previsões no conjunto de teste
pred <- predict(model, dtest)

# Calcular a precisão do modelo
accuracy <- sum(pred > 0.5 == test_labels) / length(test_labels)
print(paste("Precisão do modelo:", accuracy))


```

## Melhorias

Que melhorias você fez no seu código? 
- Ex: refatorações, melhorias de performance, acessibilidade, etc


## Suporte

Para suporte, mande um email para rafaelhenriquegallo@gmail.com


