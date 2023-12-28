import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay


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

arvore = DecisionTreeClassifier(
    max_depth=5,
    random_state=5
)
arvore.fit(x_treino,y_treino)

print(f'Acurácia treino/validacao: {arvore.score(x_treino,y_treino)}/{arvore.score(x_validacao,y_validacao)}')

y_previsto = arvore.predict(x_validacao)
matrix_confusao = confusion_matrix(y_validacao,y_previsto)

visualizacao = ConfusionMatrixDisplay(
    confusion_matrix=matrix_confusao,
    display_labels=['normal','diabete']
)
visualizacao.plot()