import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_validate,LeaveOneOut
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,classification_report,RocCurveDisplay,PrecisionRecallDisplay


def escolher_metrica():
    metrica = int(input('Qual metrica deseja analisar? 1 - Acurácia; 2 - Precisão; 3 - Recall; 4 - F1 Score:  '))
    return detectar_erros_metrica_em_valor(metrica)
    
    

def detectar_erros_metrica_em_valor(valor):
    if str(valor) not in '1234':
        raise ValueError('O valor digitado não esta entre os possíveis escolhidos!')
    return transformar_valor_em_metrica(valor)


def transformar_valor_em_metrica(valor):
    if valor == 1:
        return 'accuracy'
    elif valor == 2:
        return 'precision'
    elif valor == 3:
        return 'recall'
    elif valor == 4:
        return 'f1'


def gerar_intervalo_confianca(resultados):
    media = resultados['test_score'].mean()
    desvio_padrao = resultados['test_score'].std()
    print(f'Intervalo de confiaça: [{media-2*desvio_padrao},{min(media+2*desvio_padrao,1)}]')


dados = pd.read_csv('Aulas/parte_cinco/arquivos/emp_automovel.csv')
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

y_previsto = arvore.predict(x_val)
matrix_confusao = confusion_matrix(y_val,y_previsto)

visualizacao = ConfusionMatrixDisplay(confusion_matrix=matrix_confusao,display_labels=['adimplente','inadimplente'])
visualizacao.plot()

print(f'Acurácia: {accuracy_score(y_val,y_previsto)}')
print(f'Precisão: {precision_score(y_val,y_previsto)}')
print(f'Recall: {recall_score(y_val,y_previsto)}')
print(f'F1 Score: {f1_score(y_val,y_previsto)}')
print(f'AUC Score: {roc_auc_score(y_val,y_previsto)}')
print(f'AP: {average_precision_score(y_val,y_previsto)}')
print(classification_report(y_val,y_previsto))

RocCurveDisplay.from_predictions(y_val,y_previsto,name='Árvore de Decisão')
PrecisionRecallDisplay.from_predictions(y_val,y_previsto,name='Árvore de Decisão')


stf = StratifiedKFold(n_splits=5,shuffle=True,random_state=5)
cv_resultados = cross_validate(arvore,x,y,cv=stf)

print(cv_resultados['test_score'].mean())