import pandas as pd 
from sklearn.model_selection import train_test_split,cross_validate,StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score,confusion_matrix,classification_report
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.pipeline import Pipeline as imbpipeline


def gerar_intervalo_confianca(resultados):
    media = resultados['test_score'].mean()
    desvio_padrao = resultados['test_score'].std()
    print(media-2*desvio_padrao,min(media+2*desvio_padrao,1))


dados = pd.read_csv('Exercicios/ex012/arquivos/diabetes.csv')
x = dados.drop('diabete',axis=1)
y = dados['diabete']

x,x_teste,y,y_teste = train_test_split(
    x,y,
    stratify=y,
    test_size=0.15,
    random_state=5
)

arvore = DecisionTreeClassifier(max_depth=10)
arvore.fit(x,y)


y_previsto = arvore.predict(x_teste)
matrix_confusao = confusion_matrix(y_teste,y_previsto)

oversample = SMOTE()
x_bal,y_bal = oversample.fit_resample(x,y)

pipeline = imbpipeline([('oversample',SMOTE()),('arvore',arvore)])

skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=5)
cv_resultados = cross_validate(pipeline,x,y,cv=skf,scoring='recall')
gerar_intervalo_confianca(cv_resultados)
print(arvore.score(x_teste,y_teste))
