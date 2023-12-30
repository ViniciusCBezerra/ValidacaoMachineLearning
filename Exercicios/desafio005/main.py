import pandas as pd 
from sklearn.model_selection import train_test_split,cross_validate,KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,RocCurveDisplay,PrecisionRecallDisplay,classification_report


def gerar_intervalo_confianca(resultados):
    media = resultados['test_score'].mean()
    desvio_padrao = resultados['test_score'].std()
    print(f'Intervalo de confiança: [{media - 2*desvio_padrao},{min(media + 2*desvio_padrao,1)}]')


def reconhecer_metrica(valor):
    if valor == '1':
        return 'accuracy'
    elif valor == '2':
        return 'recall'
    elif valor == 3:
        return 'precision'
    elif valor == 4:
        return 'f1'


dados = pd.read_csv('Exercicios/desafio005/arquivos/emp_automovel.csv')
x = dados.drop('inadimplente',axis=1)
y = dados['inadimplente']


x,x_teste,y,y_teste = train_test_split(
    x,y,
    stratify=y,
    random_state=5,
    test_size=0.15
)

x_treino,x_val,y_treino,y_val = train_test_split(
    x,y,
    stratify=y,
    random_state=5,
)

arvore = DecisionTreeClassifier(
    max_depth=10,
    random_state=5
)
arvore.fit(x_treino,y_treino)

y_previsto = arvore.predict(x_val)
matrix_confusao = confusion_matrix(y_val,y_previsto)

visualizacao = ConfusionMatrixDisplay(confusion_matrix=matrix_confusao,display_labels=['adimplente','inadimplente'])
visualizacao.plot()

print(f'Acurácia: {accuracy_score(y_val,y_previsto)}')
print(f'Precisão: {precision_score(y_val,y_previsto)}')
print(f'Recall: {recall_score(y_val,y_previsto)}')
print(f'F1 Score: {f1_score(y_val,y_previsto)}')
print(f'AUC: {roc_auc_score(y_val,y_previsto)}')
print(f'AP: {average_precision_score(y_val,y_previsto)}')
print(classification_report(y_val,y_previsto))

RocCurveDisplay.from_predictions(y_val,y_previsto,name='Árvore de Decisão')
PrecisionRecallDisplay.from_predictions(y_val,y_previsto,name='Árvore de Decisão')

while True:
    metrica = int(input('Qual metrica deseja: 1 - Acurácia; 2 - Recall; 3 - Precisão; 4 - F1 Score'))
    metrica = str(metrica)
    if metrica not in '1234':
        raise ValueError('Valor digitado não está entre 1 e 4!')
    else:
        break


kf = KFold(n_splits=5,shuffle=True,random_state=5)
cv_resultados = cross_validate(arvore,x,y,cv=kf,scoring=reconhecer_metrica(metrica))
gerar_intervalo_confianca(cv_resultados)
