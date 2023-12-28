import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,RocCurveDisplay


dados = pd.read_csv('Exercicios/ex005/arquivos/emp_automovel.csv')
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
    random_state=5,
    test_size=0.15
)

arvore = DecisionTreeClassifier(
    max_depth=10,
    random_state=5
)
arvore.fit(x_treino,y_treino)

y_previsto = arvore.predict(x_validacao)
matrix_confusao = confusion_matrix(y_validacao,y_previsto)

visualizacao = ConfusionMatrixDisplay(confusion_matrix=matrix_confusao,display_labels=['adimplente','inadimplente'])
visualizacao.plot()

print(f'Acurácia: {accuracy_score(y_validacao,y_previsto)}')
print(f'Precisão: {precision_score(y_validacao,y_previsto)}')
print(f'Recall: {recall_score(y_validacao,y_previsto)}')
print(f'F1 Score: {f1_score(y_validacao,y_previsto)}')
print(f'AUC: {roc_auc_score(y_validacao,y_previsto)}')

RocCurveDisplay.from_predictions(y_validacao,y_previsto,name='Árvore de Decisão')
