# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 11:31:38 2026

@author: silva
"""

# In[0.1] Instalação dos pacotes

# pip install pandas
# pip install numpy
# pip install -U seaborn
# pip install matplotlib
# pip install plotly
# pip install scipy
# pip install statsmodels
# pip install scikit-learn
# pip install --upgrade statstests

# In[0.2] Importação dos pacotes 

import pandas as pd # manipulação de dados em formato de dataframe
import numpy as np # operações matemáticas
import seaborn as sns # visualização gráfica
import matplotlib.pyplot as plt # visualização gráfica
from scipy.interpolate import UnivariateSpline # curva sigmoide suavizada
import statsmodels.api as sm # estimação de modelos
from statstests.process import stepwise # procedimento Stepwise
from scipy import stats # estatística chi2
import plotly.graph_objects as go # gráficos 3D
from statsmodels.iolib.summary2 import summary_col # comparação entre modelos
from statsmodels.discrete.discrete_model import MNLogit # estimação do modelo
                                                        #logístico multinomial

import warnings
warnings.filterwarnings('ignore') 

# In[0.3] subindo a base de dados
df_modelo_credito = pd.read_csv('C:/Users/silva/Desktop/mod_log_binaria/base_credito_varejo.csv',delimiter=',')
df_modelo_credito

# In[0.4] caracteristicas das variaveis do dataset

df_modelo_credito.info()


# Estatísticas univariadas

df_modelo_credito.describe()

# tabela de frequências absolutas da variavel

df_modelo_credito['aprovado'].value_counts().sort_index()
df_modelo_credito['possui_cartao_proprio'].value_counts().sort_index()
df_modelo_credito['inadimplente_6m'].value_counts().sort_index()

# In[0.5] Fazendo o tratamento das variaveis dummies

df_credito_dummies = pd.get_dummies(df_modelo_credito,
                                       columns=['possui_cartao_proprio',
                                                'inadimplente_6m'],
                                       dtype=int,
                                       drop_first=True)

df_credito_dummies.info()

# In[0.5.1] Diagnóstico de Multicolinearidade (VIF e Tolerance)
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 1. Isolando as variáveis preditoras (removendo o target 'aprovado')
X_vif = df_credito_dummies.drop(columns=['aprovado'])

# 2. O cálculo do VIF exige explicitamente a presença do intercepto (constante)
X_vif = sm.add_constant(X_vif)

# 3. Criando as listas para armazenar os cálculos
vif_valores = []
tolerance_valores = []

for i in range(X_vif.shape[1]):
    vif = variance_inflation_factor(X_vif.values, i)
    vif_valores.append(vif)
    tolerance_valores.append(1 / vif if vif != 0 else 0)

# 4. Estruturando os resultados em um DataFrame para visualização
df_multicolinearidade = pd.DataFrame({
    'Variável': X_vif.columns,
    'VIF': vif_valores,
    'Tolerance': tolerance_valores
})

# 5. Filtrando a constante e ordenando do pior para o melhor caso de colinearidade
df_multicolinearidade = (df_multicolinearidade[df_multicolinearidade['Variável'] != 'const']
                         .sort_values(by='VIF', ascending=False))

# 6. Exibindo o diagnóstico formatado
print("-" * 60)
print("     DIAGNÓSTICO DE MULTICOLINEARIDADE (VIF & TOLERANCE)")
print("-" * 60)
print(df_multicolinearidade.to_string(index=False, formatters={'VIF': '{:.4f}'.format, 'Tolerance': '{:.4f}'.format}))
print("-" * 60) 

# In[0.6] Estimação do modelo completo (Ajustado)

# AJUSTE: Removido apenas o target ('aprovado'). 
# Renda, valor solicitado e comprometimento agora vão para o modelo!
lista_colunas = list(df_credito_dummies.drop(columns=['aprovado']).columns)

formula_dummies_modelo = ' + '.join(lista_colunas)
formula_dummies_modelo = "aprovado ~ " + formula_dummies_modelo
print("Fórmula utilizada: ", formula_dummies_modelo)

# Modelo propriamente dito com todas as variáveis explicativas
modelo_credito = sm.Logit.from_formula(formula_dummies_modelo,
                                       df_credito_dummies).fit(method='bfgs', maxiter=100)

# Parâmetros do 'modelo_credito' para análise de p-valor antes do stepwise
modelo_credito.summary()

# In[0.6.1] 

# Teste de separação perfeita nas categóricas
print("--- Cruzamento Inadimplência 6m ---")
print(pd.crosstab(df_credito_dummies['inadimplente_6m_1'], df_credito_dummies['aprovado']))

print("\n--- Cruzamento Cartão Próprio ---")
print(pd.crosstab(df_credito_dummies['possui_cartao_proprio_1'], df_credito_dummies['aprovado']))

# In[0.6.2] Estimação do modelo completo (Ajustado pós-Separação Perfeita)

# Retirado o target ('aprovado') e a variável que causou a separação perfeita ('inadimplente_6m_1')
variaveis_para_remover = ['aprovado', 'inadimplente_6m_1']

lista_colunas = list(df_credito_dummies.drop(columns=variaveis_para_remover, errors='ignore').columns)
formula_dummies_modelo = ' + '.join(lista_colunas)
formula_dummies_modelo = "aprovado ~ " + formula_dummies_modelo

print("Fórmula utilizada: ", formula_dummies_modelo)

# Modelo usando o método padrão, que agora vai convergir perfeitamente
modelo_credito = sm.Logit.from_formula(formula_dummies_modelo, df_credito_dummies).fit()

# Exibindo os parâmetros reais, erros padrões e p-valores para o Stepwise
print(modelo_credito.summary())

# In[0.7] Procedimento Stepwise

from statstests.process import stepwise


step_modelo_credito = stepwise(modelo_credito, pvalue_limit=0.05)

step_modelo_credito.summary()

# In[0.8] Construção da matriz confusão

from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, recall_score, f1_score

def matriz_confusao(predicts, observado, cutoff):
    
    values = predicts.values
    predicao_binaria = []
        
    for item in values:
        if item < cutoff:
            predicao_binaria.append(0)
        else:
            predicao_binaria.append(1)
           
    # Ordem padrão do scikit-learn (y_true, y_pred)
    cm = confusion_matrix(observado, predicao_binaria)
    
    # Plotagem limpa seguindo a convenção da biblioteca
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Reprovado (0)', 'Aprovado (1)'])
    disp.plot(cmap='Purples', values_format='d')
    
    plt.title(f'Matriz de Confusão (Cutoff: {cutoff})', fontsize=14)
    plt.xlabel('Predito pelo Modelo (Classified)', fontsize=12)
    plt.ylabel('Real da Base (True)', fontsize=12)
    plt.show()
        
    
    sensitividade = recall_score(observado, predicao_binaria, pos_label=1)
    especificidade = recall_score(observado, predicao_binaria, pos_label=0)
    acuracia = accuracy_score(observado, predicao_binaria)
    f1 = f1_score(observado, predicao_binaria)

    # Visualizando os principais indicadores desta matriz de confusão
    indicadores = pd.DataFrame({'Sensitividade': [sensitividade],
                                'Especificidade': [especificidade],
                                'Acurácia': [acuracia],
                                'F1-Score': [f1]})

    return indicadores

# Gera as probabilidades do modelo otimizado (Stepwise)
df_credito_dummies['phat'] = step_modelo_credito.predict()

# Executa a função corrigida e exibe a tabela com métricas reais
df_metricas = matriz_confusao(predicts=df_credito_dummies['phat'],
                              observado=df_credito_dummies['aprovado'],
                              cutoff=0.50)

display(df_metricas)



# In[0.9] Construção da curva roc

from sklearn.metrics import roc_curve, auc

# Cálculo das taxas e thresholds usando o target real e as probabilidades (phat)
fpr, tpr, thresholds = roc_curve(df_credito_dummies['aprovado'],
                                 df_credito_dummies['phat'])
roc_auc = auc(fpr, tpr)

# Cálculo do coeficiente de GINI
gini = (roc_auc - 0.5) / 0.5
plt.figure(figsize=(10, 7)) 
plt.plot(fpr, tpr, color='darkorchid', linewidth=3, label=f'Modelo Logístico (AUC = {round(roc_auc, 4)})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='dashed', label='Modelo Aleatório (AUC = 0.5000)')

# Títulos e Legendas
plt.title('Curva ROC & Coeficiente de GINI', fontsize=16, fontweight='bold', pad=25)
plt.suptitle(f'Área abaixo da curva (AUC): {round(roc_auc, 4)} | Coeficiente de GINI: {round(gini, 4)}', 
             fontsize=12, y=0.92)

plt.xlabel('1 - Especificidade (Taxa de Falsos Positivos)', fontsize=12)
plt.ylabel('Sensitividade (Taxa de Verdadeiros Positivos)', fontsize=12)

plt.xticks(np.arange(0, 1.1, 0.1), fontsize=10) 
plt.yticks(np.arange(0, 1.1, 0.1), fontsize=10)

plt.grid(True, linestyle=':', alpha=0.6) 
plt.legend(loc='lower right', fontsize=11)

plt.show()


