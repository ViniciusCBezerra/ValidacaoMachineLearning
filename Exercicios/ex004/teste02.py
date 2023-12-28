import pandas as pd 
from sklearn.model_selection import train_test_split


dados = pd.read_csv('Exercicios/ex004/arquivos/diabetes.csv')
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
