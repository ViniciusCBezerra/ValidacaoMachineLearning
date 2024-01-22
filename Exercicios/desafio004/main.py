import pandas as pd 
from sklearn.model_selection import train_test_split,cross_validate,KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,RocCurveDisplay,PrecisionRecallDisplay,classification_report


dados = pd.read_csv('Exercicios/desafio004/arquivos/diabetes.csv')
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


arvore = DecisionTreeClassifier(
    max_depth=3,
    random_state=5,
)
arvore.fit(x_treino,y_treino)

print(f'Acurácia treino: {arvore.score(x_treino,y_treino)}')
print(f'Acurácia teste: {arvore.score(x_teste,y_teste)}')

y_previsto = arvore.predict(x_validacao)
matrix_confusao = confusion_matrix(y_validacao,y_previsto)


y_previsto = arvore.predict(x_validacao)
matrix_confusao = confusion_matrix(y_validacao,y_previsto)

visualizacao = ConfusionMatrixDisplay(confusion_matrix=matrix_confusao,display_labels=['adimplente','inadimplente'])
visualizacao.plot()

print(f'Acurácia: {accuracy_score(y_validacao,y_previsto)}')
print(f'Precisão: {precision_score(y_validacao,y_previsto)}')
print(f'Recall: {recall_score(y_validacao,y_previsto)}')
print(f'F1 Score: {f1_score(y_validacao,y_previsto)}')
print(f'AUC: {roc_auc_score(y_validacao,y_previsto)}')
print(f'AP: {average_precision_score(y_validacao,y_previsto)}')

RocCurveDisplay.from_predictions(y_validacao,y_previsto,name='Árvore de Decisão')
PrecisionRecallDisplay.from_predictions(y_validacao,y_previsto,name='Árvore de Decisão')

print(classification_report(y_validacao,y_previsto))

kf = KFold(n_splits=5,shuffle=True,random_state=5)
cv_resultados = cross_validate(arvore,x,y,cv=kf)

media = cv_resultados['test_score'].mean()
desvio_padrao = cv_resultados['test_score'].std()

print(f'Intervalo de confiança [{media-2*desvio_padrao},{min(media+2*desvio_padrao,1)}]')
