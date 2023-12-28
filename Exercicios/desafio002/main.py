import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay


dados = pd.read_csv('Exercicios/desafio002/arquivos/diabetes.csv')
x = dados.drop('diabete',axis=1)
y = dados['diabete']

x,x_teste,y,y_teste = train_test_split(
    x,y,
    stratify=y,
    random_state=5,
    test_size=0.05
)

x_treino,x_validacao,y_treino,y_validacao = train_test_split(
    x,y,
    stratify=y,
    random_state=5
)

# Árvore de decisão
'''
arvore = DecisionTreeClassifier(
    max_depth=3,
    random_state=5,
)
arvore.fit(x_treino,y_treino)

print(f'Acurácia treino: {arvore.score(x_treino,y_treino)}')
print(f'Acurácia teste: {arvore.score(x_teste,y_teste)}')

y_previsto = arvore.predict(x_validacao)
matrix_confusao = confusion_matrix(y_validacao,y_previsto)
'''

# FOREST
modelo = RandomForestClassifier(
    max_depth=2,
    random_state=5
)
modelo.fit(x_treino,y_treino)

print(f'Acurácia treino: {modelo.score(x_treino,y_treino)}')
print(f'Acurácia teste: {modelo.score(x_teste,y_teste)}')

y_previsto = modelo.predict(x_validacao)
matrix_confusao = confusion_matrix(y_validacao,y_previsto)

visualizacao = ConfusionMatrixDisplay(confusion_matrix=matrix_confusao,display_labels=['adimplente','inadimplente'])
visualizacao.plot()