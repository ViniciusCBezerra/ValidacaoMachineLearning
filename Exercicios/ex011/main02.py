import pandas as pd 
from sklearn.model_selection import train_test_split,cross_validate,StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,roc_auc_score,recall_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline


def gerar_intervalo_confianca(resultados):
    media = resultados['test_score'].mean()
    desvio_padrao = resultados['test_score'].std()
    print(f'Intervalo de confiança: [{media-2*desvio_padrao},{min(media+2*desvio_padrao,1)}]')


dados = pd.read_csv('Exercicios/ex011/arquivos/emp_automovel.csv')
x = dados.drop('inadimplente',axis=1)
y = dados['inadimplente']

x,x_teste,y,y_teste = train_test_split(
    x,y,
    stratify=y,
    random_state=5,
    test_size=0.15
)
arvore = DecisionTreeClassifier(max_depth=10,random_state=5)
arvore.fit(x,y)

pipeline = imbpipeline([('oversample',SMOTE()),('arvore',arvore)])

skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=5)
cv_resultados = cross_validate(pipeline,x,y,cv=skf,scoring='recall')

print(cv_resultados['test_score'].mean())