# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 18:26:21 2024

@author: derek
"""

#%%

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

#%%

# dataframe from transparency
df_global_results_trends = pd.read_csv('/kaggle/input/global-corruption-index-transparency-perceptions/CPI2023-Global-Results-Trends.csv', delimiter='|', encoding='utf-8')

df_timeseries = pd.read_excel('/kaggle/input/global-corruption-index-transparency-perceptions/CPI2023-Global-Results-Trends.xlsx', sheet_name='CPI Timeseries 2012 - 2023', skiprows=3)

#%%

# Define a custom color scale for the choropleth map
custom_scale = [
    (0, 'rgb(255,0,0)'),  # Red for low scores
    (1, 'rgb(0,255,0)')   # Green for high scores
]

# Create a choropleth map to visualize the corruption scores by country
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

plt.figure(figsize=(12, 6))
sns.histplot(df_global_results_trends['CPI score 2023'], bins=20, kde=True, color='blue')
plt.title('Distribution of CPI Scores for 2023')
plt.xlabel('CPI Score')
plt.ylabel('Frequency')
plt.show()

#%%

top_10 = df_global_results_trends.nlargest(10, 'CPI score 2023')

plt.figure(figsize=(12, 6))
sns.barplot(x='CPI score 2023', y='Country / Territory', data=top_10, palette='viridis')
plt.title('Top 10 Countries with Highest CPI Scores in 2023')
plt.xlabel('CPI Score')
plt.ylabel('Country')
plt.show()


#%%

# Calculate correlation coefficients between CPI score and other indices
corr_cpi_cpiia = df_global_results_trends['CPI score 2023'].corr(df_global_results_trends['World Bank CPIA'])
corr_cpi_sgi = df_global_results_trends['CPI score 2023'].corr(df_global_results_trends['Bertelsmann Foundation Sustainable Governance Index'])


plt.figure(figsize=(14, 6))

# CPI score vs World Bank CPIA
plt.subplot(1, 2, 1)
sns.scatterplot(
    x='CPI score 2023', y='World Bank CPIA', 
    data=df_global_results_trends, s=100, color='blue', alpha=0.6
)
sns.regplot(
    x='CPI score 2023', y='World Bank CPIA', 
    data=df_global_results_trends, scatter=False, color='blue'
)
plt.title(f'CPI score vs World Bank CPIA (Correlation: {corr_cpi_cpiia:.2f})')
plt.xlabel('CPI score 2023')
plt.ylabel('World Bank CPIA')

# CPI score vs Bertelsmann Foundation Sustainable Governance Index
plt.subplot(1, 2, 2)
sns.scatterplot(
    x='CPI score 2023', y='Bertelsmann Foundation Sustainable Governance Index', 
    data=df_global_results_trends, s=100, color='green', alpha=0.6
)
sns.regplot(
    x='CPI score 2023', y='Bertelsmann Foundation Sustainable Governance Index', 
    data=df_global_results_trends, scatter=False, color='green'
)
plt.title(f'CPI score vs Bertelsmann Governance Index (Correlation: {corr_cpi_sgi:.2f})')
plt.xlabel('CPI score 2023')
plt.ylabel('Bertelsmann Foundation Sustainable Governance Index')

plt.tight_layout()
plt.show()

#%%

#Corrução e Governança

# Select relevant columns for analysis
columns_of_interest = ['CPI score 2023', 'Varieties of Democracy Project', 'Freedom House Nations in Transit']
df_subset = df_global_results_trends[columns_of_interest]

# Calculate the correlation matrix
correlation_matrix = df_subset.corr()

plt.figure(figsize=(10, 8))

# Create a heatmap to visualize the correlation matrix with larger annotations
sns.heatmap(
    correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f",
    annot_kws={"size": 14}, linewidths=.5
)
plt.title('Correlation Heatmap', fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(rotation=0, fontsize=12)

# Display the plot
plt.show()

#%%

#Tendências de pontuação CPI

# CPI Score Trends (2012-2023) with better visualization without outliers
# Filter out the data to include only the relevant columns for the trends
df_trends_filtered = df_timeseries[['Country / Territory', 'ISO3', 'Region'] + [col for col in df_timeseries.columns if 'CPI score' in col]]

# Melt the dataframe for easier plotting
df_trends_melted = df_trends_filtered.melt(id_vars=['Country / Territory', 'ISO3', 'Region'], 
                                           var_name='Year', 
                                           value_name='CPI Score')

# Extract the year from the column names
df_trends_melted['Year'] = df_trends_melted['Year'].str.extract('(\d{4})').astype(int)

# Filter the data to remove outliers and only include relevant countries for trend analysis
selected_countries = ['Denmark', 'Finland', 'New Zealand', 'Norway', 'Singapore', 'Somalia', 'Syria', 'South Sudan']
df_trends_selected = df_trends_melted[df_trends_melted['Country / Territory'].isin(selected_countries)]

plt.figure(figsize=(14, 8))
sns.lineplot(data=df_trends_selected, x='Year', y='CPI Score', hue='Country / Territory', marker='o')

plt.title('CPI Score Trends (2012-2023) for Selected Countries')
plt.xlabel('Year')
plt.ylabel('CPI Score')
plt.ylim(0, 100)  # Setting the y-axis limit between 0 and 100
plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

#%%

#Comparação das puntuações

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