import pandas as pd 
from sklearn.model_selection import train_test_split,cross_validate,StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from imblearn.under_sampling import ClusterCentroids


def gerar_intervalo_confianca(resultados):
    media = resultados['test_score'].mean()
    desvio_padrao = resultados['test_score'].std()
    print(f'Intervalo de Confiança: [{media-2*desvio_padrao},{min(media+2*desvio_padrao,1)}]')


dados = pd.read_csv('Exercicios/ex010/arquivos/emp_automovel.csv')
x = dados.drop('inadimplente',axis=1)
y = dados['inadimplente']

x,x_teste,y,y_teste = train_test_split(
    x,y,
    stratify=y,
    test_size=0.15,
    random_state=5
)
x_treino,x_val,y_treino,y_val = train_test_split(
    x,y,
    stratify=y,
    test_size=0.15,
    random_state=5
)
arvore = DecisionTreeClassifier(max_depth=10)
arvore.fit(x_treino,y_treino)

skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=5)
cv_resultados = cross_validate(arvore,x,y,cv=skf,scoring='recall')
gerar_intervalo_confianca(cv_resultados)

undersample = ClusterCentroids()
x_bal,y_bal = undersample.fit_resample(x,y)
skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=5)
cv_resultados = cross_validate(arvore,x_bal,y_bal,cv=skf,scoring='recall')
gerar_intervalo_confianca(cv_resultados)
