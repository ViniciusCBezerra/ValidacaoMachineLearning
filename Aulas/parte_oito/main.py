import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,cross_validate,KFold,StratifiedKFold
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,accuracy_score,precision_score,recall_score,f1_score,RocCurveDisplay,roc_auc_score,PrecisionRecallDisplay,average_precision_score,classification_report
from imblearn.under_sampling import NearMiss
from imblearn.pipeline import Pipeline as imbpipeline


def intervalo_conf(resultados):
    media = resultados['test_score'].mean()
    desvio_padrao = resultados['test_score'].std()
    print(media-2*desvio_padrao,min(media+2*desvio_padrao,1))


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

'''
print(f'Acurácia: {accuracy_score(y_val,y_previsto)}')
print(f'Precisão: {precision_score(y_val,y_previsto)}')
print(f'Recall: {recall_score(y_val,y_previsto)}')
print(f'F1 Score: {f1_score(y_val,y_previsto)}')
print(f'AUC Score: {roc_auc_score(y_val,y_previsto)}')
print(f'AP: {average_precision_score(y_val,y_previsto)}')
print(classification_report(y_val,y_previsto))
'''

RocCurveDisplay.from_predictions(y_val,y_previsto,name='Árvore de Decisão')
PrecisionRecallDisplay.from_predictions(y_val,y_previsto,name='Árvore de Decisão')

pipeline = imbpipeline([('undersample',NearMiss(version=3)),('arvore',arvore)])

skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=5)
cv_resultados = cross_validate(pipeline,x, y, cv = skf,scoring='recall')

undersample = NearMiss(version=3)
x_balanceado,y_balanceado = undersample.fit_resample(x,y)
modelo = DecisionTreeClassifier(max_depth=10)
modelo.fit(x_balanceado,y_balanceado)
y_previsto = modelo.predict(x_teste)

print(classification_report(y_teste,y_previsto))
print(confusion_matrix(y_teste,y_previsto))
