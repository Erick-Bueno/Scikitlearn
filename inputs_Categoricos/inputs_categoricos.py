import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import numpy as np


dados = pd.read_csv("exemplo3.csv")
X = dados.drop('risco', axis=1)
y = dados.risco
#X.select_dtypes(include=object)selecionando dados categoricos no caso do exemplo o sexo, exclude = apenas colunas numericas

X_categ = X.select_dtypes(include=object)

#converter o dado categorico(sexo) em binario
onehot = OneHotEncoder( sparse=False, drop='first')
X_bin = onehot.fit_transform(X_categ)

#pegando apenas idade e conta corrente
X_num = X.select_dtypes(exclude=object)

#Normalizador(converter os dados não categoricos)
Normalizador = MinMaxScaler()
X_Norm = Normalizador.fit_transform(X_num)


#unir os dados categoricos com os numericos ambos ja convertidos 
X_junto = np.append(X_Norm, X_bin, axis=1 )
print(X_junto)