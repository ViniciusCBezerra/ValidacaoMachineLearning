import pandas as pd 
from sklearn.model_selection import train_test_split,cross_validate,KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,RocCurveDisplay,PrecisionRecallDisplay,classification_report


def gerar_intervalo_confianca(resultado):
    media = resultado['test_score'].mean()
    desvio_padrao = resultado['test_score'].std()
    print(f'Intervalo de confiança: [{media-2*desvio_padrao},{min(media+2*desvio_padrao,1)}]')


def retornar_metrica():
    metrica = pegar_metrica()
    return reconhecer_metrica(metrica) 


def pegar_metrica():
    metrica = int(input('Qual metrica deseja? 1 - Acurácia; 2 - Precisão; 3 - Recall; 4 - F1 Score:  '))
    validar_valor(metrica)
    return metrica


def validar_valor(valor):
    if valor >= 5 or valor <= 0:
        raise ValueError('O valor é invalido! Ele não está na lista de metricas!')


def reconhecer_metrica(valor):
    if valor == 1:
        metrica = 'accuracy'
    elif valor == 2:
        metrica = 'precision'
    elif valor == 3:
        metrica = 'recall'
    elif valor == 4:
        metrica = 'f1'
    return metrica


dados = pd.read_csv('Exercicios/ex006/arquivos/emp_automovel.csv')
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

metrica = retornar_metrica()

kf = KFold(n_splits=5,shuffle=True,random_state=5)
cv_resultado = cross_validate(arvore,x,y,cv=kf,scoring=metrica)

gerar_intervalo_confianca(cv_resultado)
