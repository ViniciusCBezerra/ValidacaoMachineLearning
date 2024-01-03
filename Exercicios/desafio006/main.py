import pandas as pd 
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold,LeaveOneOut,cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay


def gerar_intervalo_confianca(resultado):
    media = resultado.mean()
    desvio_padrao = resultado.std()
    print(f'Intervalo de Confiança: [{media-2*desvio_padrao},{min(media+2*desvio_padrao,1)}]')


dados = pd.read_csv('Exercicios/desafio006/arquivos/diabetes.csv')
x = dados.drop('diabete',axis=1)
y = dados['diabete']

x,x_teste,y,y_teste = train_test_split(
    x,y,
    stratify=y,
    test_size=0.15,
    random_state=5
)

x_treino,x_validacao,y_treino,y_validacao = train_test_split(
    x,y,
    stratify=y,
    random_state=5
)

arvore = DecisionTreeClassifier(max_depth=10)
arvore.fit(x_treino,y_treino)

y_previsto = arvore.predict(x_validacao)
matrix_confusao = confusion_matrix(y_validacao,y_previsto)

visualizacao = ConfusionMatrixDisplay(confusion_matrix=matrix_confusao,display_labels=['não','sim'])
visualizacao.plot()

kf = KFold(n_splits=10,shuffle=True,random_state=5)
cv_resultado_kf = cross_val_score(arvore,x,y,cv=kf)
gerar_intervalo_confianca(cv_resultado_kf)


skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=5)
cv_resultado_skf = cross_val_score(arvore,x,y,cv=skf,scoring='f1')
gerar_intervalo_confianca(cv_resultado_skf)


loo = LeaveOneOut()
cv_resultado_loo = cross_val_score(arvore,x,y,cv=loo)
media_loo = cv_resultado_loo.mean()
print(media_loo)
