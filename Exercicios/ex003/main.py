import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

dados = pd.read_csv('Exercicios/ex003/arquivos/diabetes.csv')
x = dados.drop('diabete',axis=1)
y = dados['diabete']

x,x_teste,y,y_teste = train_test_split(
    x,y,
    stratify=y,
    random_state=5,
    test_size=0.15
)

x_treino,x_validacao,y_treino,y_validacao = train_test_split(
    x,y,
    stratify=y,
    random_state=5
)

# Arvore de Decisao
'''
arvore = DecisionTreeClassifier(
    max_depth=5,
    random_state=5
)
arvore.fit(x_treino,y_treino)
print(arvore.score(x_validacao,y_validacao))

y_previsto = arvore.predict(x_validacao)
matrix_confusao = confusion_matrix(y_validacao,y_previsto)
print(matrix_confusao)
'''

# FOREST

modelo = RandomForestClassifier(
    max_depth=3,
    random_state=5
)

modelo.fit(x_treino,y_treino)
print(modelo.score(x_validacao,y_validacao))

y_previsto = modelo.predict(x_validacao)
matrix_confusao = confusion_matrix(y_validacao,y_previsto)
print(matrix_confusao)
