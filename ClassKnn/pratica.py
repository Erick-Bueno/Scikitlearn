import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#1 ler os dados
dados = pd.read_csv('exemplo2.csv')
#2-separar os inputs e output
X = dados.drop('risco', axis=1)
y = dados.risco
#3-normalizador(melhorando a precisão do classificador)
Normalizador = MinMaxScaler()
#3.1 - converter os dados de input para um valor na escala 0 e 1 para que os dados na hora da analise tenham a mesma prioridade
New_X = Normalizador.fit_transform(X)

#4-separando em subconjuntos
X_train, X_test, y_train, y_test = train_test_split(New_X, y, train_size=2/3)
#5-treinamento
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
#5-teste
accuracy_score(y_test, knn.predict(X_test))
#6-previsão
novo_cliente = [[18,500],[22,700]]
X_newClient = Normalizador.transform(novo_cliente)
print(knn.predict(X_newClient))

