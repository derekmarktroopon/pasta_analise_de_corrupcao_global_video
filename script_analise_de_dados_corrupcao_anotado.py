# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 18:26:21 2024

@author: derek
"""

#%%

#IMPORTAÇÃO DOS PACOTES NECESSÁRIOS

# Import necessary libraries for data analysis and visualization
import numpy as np
import pandas as pd

# Visualization libraries
!pip install plotly==4.14.3
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical and machine learning libraries
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# Plotly settings
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)

#Usado para criar o modelo de regressão linear multivariável
import statsmodels.api as sm

#%%

#IMPORTAÇÃO DOS DADOS

# dataframe from transparency
df_global_results_trends = pd.read_csv('C:/Users/derek/OneDrive/Documentos/pasta_analise_de_corrupcao_global_video/CPI2023-Global-Results-Trends.csv', delimiter='|', encoding='utf-8')

df_timeseries = pd.read_excel('C:/Users/derek/OneDrive/Documentos/pasta_analise_de_corrupcao_global_video/CPI2023-Global-Results-Trends.xlsx', sheet_name='CPI Timeseries 2012 - 2023', skiprows=3)

#%%

#DEFINIÇÃO DA MAPA 

# Definição da grade de cor para a mapa (vermelho para pontuação baixa e verde para pontuação alta)
custom_scale = [
    (0, 'rgb(255,0,0)'),  # Red for low scores
    (1, 'rgb(0,255,0)')   # Green for high scores
]

# Criação da mapa cloropleth da grade de pontuação CPI por país
fig = px.choropleth(
    df_global_results_trends, 
    locations="ISO3",
    color="CPI score 2023",
    hover_name="Country / Territory",
    color_continuous_scale=custom_scale,
    range_color=(0, 100)
)

# Update layout of the plot
fig.update_layout(title='Corruption score by countries in 2023')

# Display the plot
fig.show()

#%%

#CRIAÇÃO DE UMA HISTOGRAMA DA DISTRIBUIÇÃO DA FREQUÊNCIA DE PONTUAÇÃO CPI PAR 2023

plt.figure(figsize=(12, 6))
sns.histplot(df_global_results_trends['CPI score 2023'], bins=20, kde=True, color='blue')
plt.title('Distribution of CPI Scores for 2023')
plt.xlabel('CPI Score')
plt.ylabel('Frequency')
plt.show()

#%%

#GRÁFICO - 10 PAÍSES COM A PONTUAÇÃO CPI MAIORES

top_10 = df_global_results_trends.nlargest(10, 'CPI score 2023')

plt.figure(figsize=(12, 6))
sns.barplot(x='CPI score 2023', y='Country / Territory', data=top_10, palette='viridis')
plt.title('Top 10 Countries with Highest CPI Scores in 2023')
plt.xlabel('CPI Score')
plt.ylabel('Country')
plt.show()


#%%

#GRÁFICO - 10 PAÍSES COM A PONTUAÇÃO CPI MENORES

bottom_10 = df_global_results_trends.nsmallest(10, 'CPI score 2023')

plt.figure(figsize=(12, 6))
sns.barplot(x='CPI score 2023', y='Country / Territory', data=bottom_10, palette='magma')
plt.title('Bottom 10 Countries with Lowest CPI Scores in 2023')
plt.xlabel('CPI Score')
plt.ylabel('Country')
plt.show()

#%%



# CALCULO DA CORRELAÇÃO ENTRE A CPI E INDICES DE ESTABLILIDADE GOVERNAMENTAL (O WORLD BANK CPIA; E O BERTELSMANN FOUNDATION SUSTAINABLE GOVERNANCE INDEX)
corr_cpi_cpiia = df_global_results_trends['CPI score 2023'].corr(df_global_results_trends['World Bank CPIA'])
corr_cpi_sgi = df_global_results_trends['CPI score 2023'].corr(df_global_results_trends['Bertelsmann Foundation Sustainable Governance Index'])


# GRAFÍCO DE DISTRIBUIÇÃO COM FUNÇÃO DE CORRELAÇÃO ENTRE CPI E CPIA (REGRESSÃO LINEAR)
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(
    x='CPI score 2023', y='World Bank CPIA', 
    data=df_global_results_trends, s=100, color='blue', alpha=0.6
)
sns.regplot( #syntax de criaçaõ de função de regressão linear, que inclui a margem de erro
    x='CPI score 2023', y='World Bank CPIA', 
    data=df_global_results_trends, scatter=False, color='blue'
)
plt.title(f'CPI score vs World Bank CPIA (Correlation: {corr_cpi_cpiia:.2f})')
plt.xlabel('CPI score 2023')
plt.ylabel('World Bank CPIA')

plt.figure(figsize=(14, 6))

# GRAFÍCO DE DISTRIBUIÇÃO COM FUNÇÃO DE CORRELAÇÃO ENTRE CPI E BFSGI (REGRESSÃO LINEAR)
plt.subplot(1, 2, 2)
sns.scatterplot(
    x='CPI score 2023', y='Bertelsmann Foundation Sustainable Governance Index', 
    data=df_global_results_trends, s=100, color='green', alpha=0.6
)
sns.regplot( #syntax de criaçaõ de função de regressão linear, que inclui a margem de erro
    x='CPI score 2023', y='Bertelsmann Foundation Sustainable Governance Index', 
    data=df_global_results_trends, scatter=False, color='green'
)
plt.title(f'CPI score vs Bertelsmann Governance Index (Correlation: {corr_cpi_sgi:.2f})')
plt.xlabel('CPI score 2023')
plt.ylabel('Bertelsmann Foundation Sustainable Governance Index')

plt.tight_layout()
plt.show()

#%%

#CALCULO DAS CORRELAÇÕES ENTRE CORRUPÇÃO E LIBERDADE/DEMOCRACIA

# Select relevant columns for analysis
columns_of_interest = ['CPI score 2023', 'Varieties of Democracy Project', 'Freedom House Nations in Transit']
df_subset = df_global_results_trends[columns_of_interest]


# Criação de uma mapa de calor (matriz de correlações)
correlation_matrix = df_subset.corr()

plt.figure(figsize=(10, 8))

# montagem do mapa de calor das coeficientes corrupção x liberdades
sns.heatmap(
    correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f",
    annot_kws={"size": 14}, linewidths=.5
)
plt.title('Correlation Heatmap', fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(rotation=0, fontsize=12)

# Display the plot
plt.show()

#MOSTRA DQUE A CORRELAÇÃO ENTRE CORRUPÇÃO E LIBERDADES / DEMOCRACIA SÃO QUASE 1 AO 1

#%%

#AS TENDÊNCIAS DE PONTUAÇÃO CPI 

# Preparação dos dados para mostrar so a pontuação do CPI para anos fiferentes
df_trends_filtered = df_timeseries[['Country / Territory', 'ISO3', 'Region'] + [col for col in df_timeseries.columns if 'CPI score' in col]]

# 'melting' os dados (compactando de uma forma melhor) para facilitar a plotagem
df_trends_melted = df_trends_filtered.melt(id_vars=['Country / Territory', 'ISO3', 'Region'], 
                                           var_name='Year', 
                                           value_name='CPI Score')

# tirando os anos a partir das nomes das colunas, para usar como rótulos
df_trends_melted['Year'] = df_trends_melted['Year'].str.extract('(\d{4})').astype(int)

# filtrando os dados (tirando os outliers)
selected_countries = ['Denmark', 'Finland', 'New Zealand', 'Norway', 'Singapore', 'Somalia', 'Syria', 'South Sudan']
df_trends_selected = df_trends_melted[df_trends_melted['Country / Territory'].isin(selected_countries)]

#plotágem do gráfico
plt.figure(figsize=(14, 8))
sns.lineplot(data=df_trends_selected, x='Year', y='CPI Score', hue='Country / Territory', marker='o')

plt.title('CPI Score Trends (2012-2023) for Selected Countries')
plt.xlabel('Year')
plt.ylabel('CPI Score')
plt.ylim(0, 100)  # Setting the y-axis limit between 0 and 100
plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

#%%

#Comparação das puntuações médias entre indíces diferentes de corrupção 

sources_cols = ['African Development Bank CPIA', 'Bertelsmann Foundation Sustainable Governance Index',
                'Bertelsmann Foundation Transformation Index', 'Economist Intelligence Unit Country Ratings',
                'Freedom House Nations in Transit', 'Global Insights Country Risk Ratings',
                'IMD World Competitiveness Yearbook', 'PERC Asia Risk Guide',
                'PRS International Country Risk Guide', 'Varieties of Democracy Project',
                'World Bank CPIA', 'World Economic Forum EOS', 'World Justice Project Rule of Law Index']

source_means = df_global_results_trends[sources_cols].mean().reset_index()
source_means.columns = ['Source', 'Average Score']

plt.figure(figsize=(14, 8))
sns.barplot(x='Average Score', y='Source', data=source_means, palette='viridis')
plt.title('Comparison of Average Scores from Different Sources for 2023')
plt.xlabel('Average Score')
plt.ylabel('Source')
plt.show()

#%%

#ANÁLISE DAS RELAÇÕES ENTRE AS COEFICIENTES E A ÍNDICE CPI

reg = smf.ols(formula='lnwage ~ education + age + I(age**2)', data=df_global_results_trends)
results = reg.fit()
print(results.summary())

#%%

#MODELO DE ESTIMATÍVA CPI - REGRESSÃO LINEAR MULTIVARIÁVEL

#Criando uma tabéla com as variáveis de interesse - (objetívo do modelo: ver compo as níveis de estabilidade governamental e liberdade podem ser usadas para estimar a corrupção de uma país)
columns_of_interest = ['CPI score 2023', 'World Bank CPIA', 'Bertelsmann Foundation Sustainable Governance Index', 
                       'Varieties of Democracy Project', 'Freedom House Nations in Transit']
df_subset = df_global_results_trends[columns_of_interest]

# Handle missing values if any
imputer = SimpleImputer(strategy='mean')
df_subset_imputed = pd.DataFrame(imputer.fit_transform(df_subset), columns=columns_of_interest)

# Define the dependent variable (Y) and independent variables (X)
Y_cpi_estimation = df_subset_imputed['CPI score 2023']
X_cpi_estimation = df_subset_imputed[['World Bank CPIA', 'Bertelsmann Foundation Sustainable Governance Index', 
                       'Varieties of Democracy Project', 'Freedom House Nations in Transit']]

# Add a constant to the independent variables matrix for the intercept
X = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(Y, X).fit()

# Print the model summary
print(model.summary())

#ANÁLISE 

#O R quadrado do modelo é de 0,837, significando que 83,7% da variação de corrupção de um país pode ser estimado pelo seu grau de liberdade/democracia e estabilidade governamental.

#A nota F calculada é muito maior (224) do que a nota F tabelada (0,0000--[E-64]--106), isso significa que o modelo tem grande relevância estatística.

#O Variável independente com o maior relevância é a nota Var. of Democ. proj. (a nota da nível de Democracia num país).
# Aqui podemos ver uma coeficiente de 0,7233, significando que cada nível de pontuação nesse indíce de democracia aumenta a pontuação da CPI por 0,7233.
# Podemos ver que esse coeficiente é extremamente estatisticamente relevante, com um t valor calculado (23,5) muito superior ao valor t tabelado de 0.

#O segundo varável independente mais estatísticamente relevante nesse modelo é o BF Sustain. Gov. Index (a nota de governança sustentável num país).
# Aqui podemos ver uma coeficiente de 0,3127, significando que cada nível de pontuação nesse indíce de sustentabilidade de governânça aumenta a pontuação da CPI por 0,3127.
# Podemos ver que esse coeficiente é muito estatisticamente relevante, com um t valor calculado (4,624) superior ao valor t tabelado de 0.

#O terceiro varável independente mais relevante nesse modelo é o World Bank CPIA. (a nota da integridade das instituições num país).
# Aqui podemos ver uma coeficiente de 0,3475, significando que cada nível de pontuação nesse indíce da integridade de instituições aumenta a pontuação da CPI por 0,3275.
# Podemos ver que esse coeficiente é muito estatisticamente relevante, com um t valor calculado (4,609) superior ao valor t tabelado de 0.

#O varável independente final é aquele que possui o menor relevância nesse modelo, e é a nota dada pelo Freedom House Nations in Transit (mostrando os níveis de liberdade que pessoas possuem num país).
# Aqui podemos ver uma coeficiente baixa de 0,0584, significando que cada nível de pontuação nesse indíce de liberdade pessoal aumenta a pontuação da CPI por 0,0584.
# Podemos ver que esse coeficiente não é muito estatisticamente relevante, com um t valor calculado (0,58) só um pouco superior ao valor t tabelado de 0,563, 
# ainda que isso mostra que não é completamente irrelevante na cálculo da estimativa do CPI.
