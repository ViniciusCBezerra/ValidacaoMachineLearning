import pandas as pd
from sklearn.model_selection import train_test_split,cross_validate,StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,classification_report
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.pipeline import Pipeline as imbpipeline


dados = pd.read_csv('Exercicios/desafio007/arquivos/diabetes.csv')
x = dados.drop('diabete',axis=1)
y = dados['diabete']

x,x_teste,y,y_teste = train_test_split(
    x,y,
    stratify=y,
    test_size=0.15,
    random_state=5
)
modelo = DecisionTreeClassifier(max_depth=10)
oversample = SMOTE()
undersample = NearMiss(version=3)
x_balanceado,y_balanceado = oversample.fit_resample(x,y)
modelo.fit(x_balanceado,y_balanceado)

y_previsto = modelo.predict(x_balanceado)
y_teste_previsto = modelo.predict(x_teste)

print(classification_report(y_teste,y_teste_previsto))
print(confusion_matrix(y_balanceado,y_previsto))

pipeline = imbpipeline([('undersample',NearMiss(version=3)),('arvore',modelo)])

skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=5)
cv_resultados = cross_validate(pipeline,x,y,cv=skf,scoring='recall')
print(cv_resultados['test_score'].mean())
