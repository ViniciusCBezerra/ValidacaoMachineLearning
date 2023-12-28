import pandas as pd 
from sklearn.tree import DecisionTreeClassifier

dados = pd.read_csv('parte_um/arquivos/emp_automovel.csv')
x = dados.drop('inadimplente',axis=1)
y = dados['inadimplente']

arvore = DecisionTreeClassifier()
arvore.fit(x,y)
print(f'Acurácia: {arvore.score(x,y)}')
