from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd 

dados = pd.read_csv('Exercicios/desafio001/arquivos/emp_automovel.csv')
x = dados.drop('inadimplente',axis=1)
y = dados['inadimplente']

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

modelo = RandomForestClassifier(
    max_depth=3
)
modelo.fit(x_treino,y_treino)
print(f'Acurácia: {modelo.score(x_validacao,y_validacao)}')

