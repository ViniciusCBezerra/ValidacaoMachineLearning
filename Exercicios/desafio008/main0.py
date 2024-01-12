import pandas as pd
from sklearn.model_selection import train_test_split,cross_validate,StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score,classification_report,confusion_matrix
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as imbpipeline


def gerar_intervalo_confianca(resultados):
    media = resultados['test_score'].mean()
    desvio_padrao = resultados['test_score'].std()
    print(f'[{media-2*desvio_padrao},{min(media+2*desvio_padrao,1)}]')


dados = pd.read_csv('Exercicios/desafio008/arquivos/emp_automovel.csv')
x = dados.drop('inadimplente',axis=1)
y = dados['inadimplente']

x,x_teste,y,y_teste = train_test_split(
    x,y,
    stratify=y,
    test_size=0.15,
    random_state=5
)
arvore = DecisionTreeClassifier(max_depth=10)
arvore.fit(x,y)
y_previsto = arvore.predict(x)

combine = SMOTEENN()
x_bal,y_bal = combine.fit_resample(x,y)
arvore.fit(x_bal,y_bal)
y_previsto = arvore.predict(x_bal)
y_previsto_teste = arvore.predict(x_teste)

print(classification_report(y_bal,y_previsto))
print(classification_report(y_teste,y_previsto_teste))

pipeline = imbpipeline([('combine',SMOTEENN()),('modelo',arvore)])

skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=5)
cv_resultados = cross_validate(pipeline,x,y,cv=skf,scoring='recall')
gerar_intervalo_confianca(cv_resultados)
