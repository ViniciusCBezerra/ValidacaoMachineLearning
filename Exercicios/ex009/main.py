import pandas as pd 
from sklearn.model_selection import train_test_split,cross_validate,StratifiedKFold
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,RocCurveDisplay,PrecisionRecallDisplay,classification_report


def gerar_intervalo_confianca(resultado):
    media = resultado['test_score'].mean()
    desvio_padrao = resultado['test_score'].std()
    print(f'Intervalo de confiança: [{media-2*desvio_padrao},{min(media+2*desvio_padrao,1)}]')


dados = pd.read_csv('Exercicios/ex009/arquivos/emp_automovel.csv')
x = dados.drop('inadimplente',axis=1)
y = dados['inadimplente']


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

nome_colunas = [
    'receita_cliente',
    'anuidade_emprestimo',
    'anos_casa_propria',
    'telefone_trab',
    'avaliacao_cidade',
    'score_1',
    'score_2',
    'score_3',
    'score_social',
    'troca_telefone',
    'inadimplente'
]

plot_tree(arvore,filled=True,class_names=['adimplente','inalimplente'],fontsize=5,feature_names=nome_colunas)

y_previsto = arvore.predict(x_validacao)
matrix_confusao = confusion_matrix(y_validacao,y_previsto)

visualizacao = ConfusionMatrixDisplay(confusion_matrix=matrix_confusao,display_labels=['adimplente','inadimplente'])
visualizacao.plot()

print(f'Acurácia: {accuracy_score(y_validacao,y_previsto)}')
print(f'Precisão: {precision_score(y_validacao,y_previsto)}')
print(f'Recall: {recall_score(y_validacao,y_previsto)}')
print(f'F1 Score: {f1_score(y_validacao,y_previsto)}')
print(f'AUC: {roc_auc_score(y_validacao,y_previsto)}')
print(f'AP: {average_precision_score(y_validacao,y_previsto)}')
print(f'Classification Report: {classification_report(y_validacao,y_previsto)}')

RocCurveDisplay.from_predictions(y_validacao,y_previsto,name='Árvore de Decisão')
PrecisionRecallDisplay.from_predictions(y_validacao,y_previsto,name='Árvore de Decisão')

skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=5)
cv_resultados = cross_validate(arvore,x,y,cv=skf,scoring='recall')

gerar_intervalo_confianca(cv_resultados)
