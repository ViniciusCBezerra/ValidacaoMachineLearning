import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree

dados = pd.read_csv('Exercicios/ex002/arquivos/emp_automovel.csv')
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

arvore = DecisionTreeClassifier(
    max_depth=5
)
arvore.fit(x_treino,y_treino)
colunas = [
    'receita_cliente', 'anuidade_emprestimo', 'anos_casa_propria',       
    'telefone_trab', 'avaliacao_cidade', 'score_1', 'score_2', 'score_3',
    'score_social', 'troca_telefone'
]

plot_tree(arvore,filled=True,class_names=['nao','sim'],fontsize=5,feature_names=colunas)
