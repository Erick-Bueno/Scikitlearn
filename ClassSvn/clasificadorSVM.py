import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dados = pd.read_csv('exemplo2.csv')

X = dados.drop('risco', axis=1)
y = dados.risco


Normalizador = MinMaxScaler()
X_norm = Normalizador.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_norm, y,train_size=2/3)
svc= SVC()
svc.fit(X_train, y_train)



print(accuracy_score(y_test, svc.predict(X_test)))


