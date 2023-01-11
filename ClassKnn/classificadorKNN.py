from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#knn se trata de um classificador lazy pois n necessita de treinamento sua analise ja é feita diretamente, classificadores lazy não são utilizados por conta da sua desvantagem computacional pois como não treinam necessitam de um tempo maior quando vão classificar.

#lendo o arquivo
frame = pd.read_csv('exemplo2.csv')

#--------------------------------------------------------
# 1-separar oq vc quer prever doq vc vai utilizar para prever, nesse caso iremos prever o risco utilizando a idade a conta corrente

#por padrão é utilizado o X maiusculo para chamar os dados q serao utilizados para prever e o y minusculo para oq vc quer prever

X= frame.drop('risco', axis=1) #axis=1 == colunas, axis= 0 == linhas
y = frame.risco #selecionando uma coluna

#-----------------------------------------------------------
#no knn é necessario informar o tamanho da vizinhaça q nesse exemplo foi declarado como 3
knn = KNeighborsClassifier(n_neighbors=3)

#2-para o treinamento utilizamos o metodo fit
knn.fit(X, y) #não é aconsolhavel fazer o treinamento em todo os dados como nesse exemplo

#3 para fazer a previsão utilizamos predict
#preditct recebe uma tabela como parametro, [] = tabela [[],[]] = linhas na tabela
#ex 
knn.predict([[18,1000],[22,2000]])


#4 para separar os dados em conjuntos utilizamos o train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=2/3)
knn.fit(X_train, y_train)
#y_test = valores que ja foram classificados
#X_test = valores que serão classificados



#5 para calcular a precisão do classificador utilizamos o acurracy_score
print(accuracy_score(y_test, knn.predict(X_test)))
