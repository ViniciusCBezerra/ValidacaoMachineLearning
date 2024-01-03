import pandas as pd 
from sklearn.model_selection import train_test_split,cross_validate,StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE


def gerar_intervalo_confianca(resultados):
    media = resultados['test_score'].mean()
    desvio_padrao = resultados['test_score'].std()
    print(f'Intervalo de confiança: [{media-2*desvio_padrao},{min(media+2*desvio_padrao,1)}]')


dados = pd.read_csv('Exercicios/desafio006/arquivos/diabetes.csv')
x = dados.drop('diabete',axis=1)
y = dados['diabete']

x,x_teste,y,y_teste = train_test_split(
    x,y,
    stratify=y,
    random_state=5
)

arvore = DecisionTreeClassifier(max_depth=5,random_state=5)

oversample = SMOTE()
x_balanciado,y_balanciado = oversample.fit_resample(x,y)

skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=5)
cv_resultados = cross_validate(arvore,x_balanciado,y_balanciado,cv=skf,scoring='recall')

gerar_intervalo_confianca(cv_resultados)
