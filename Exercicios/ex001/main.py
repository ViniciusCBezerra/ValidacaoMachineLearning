import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

dados = pd.read_csv('Exercicios/ex001/arquivos/emp_automovel.csv')
x = dados.drop('inadimplente',axis=1)
y = dados['inadimplente']

arvore = DecisionTreeClassifier(
    max_depth=10
)

x,x_teste,y,y_teste = train_test_split(
    x,y,
    stratify=y,
    random_state=5,
    test_size=0.15
)

x_treino,x_val,y_treino,y_val = train_test_split(
    x,y,
    stratify=y,
    random_state=5
)

arvore.fit(x_treino,y_treino)

print(f'Acurácia treino: {arvore.score(x_treino,y_treino)}')
print(f'Acurácia validação: {arvore.score(x_val,y_val)}')
