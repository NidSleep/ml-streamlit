import streamlit as st
import pandas as pd
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt

# Title
st.title('Birch Clustering Visualization')
st.set_option('deprecation.showPyplotGlobalUse', False)
# Sidebar for threshold selection
threshold = st.slider('Threshold', min_value=0.1, max_value=0.5, step=0.1, value=0.1)

# Load the data from GitHub raw URL
url = 'https://raw.githubusercontent.com/NidSleep/streamlit-example/master/dataset_cleansed.csv'
df = pd.read_csv(url, encoding='utf-8')


# Selecting relevant columns and handling missing values
df_coordinates = df[['Longitude', 'Latitude', 'Fatalities', 'Injured', 'Total victims', 'Policeman Killed']]
df_coordinates.dropna(inplace=True)
df_casualty = df_coordinates[['Fatalities', 'Injured', 'Total victims', 'Policeman Killed']]

# Calculate silhouette score
birch = Birch(threshold=threshold, n_clusters=None)
cluster_labels = birch.fit_predict(df_casualty)
silhouette_score_value = silhouette_score(df_casualty, cluster_labels)

# Plot the clustering results
plt.scatter(df_casualty['Fatalities'], df_casualty['Injured'], c=cluster_labels, cmap='viridis', alpha=0.5)
plt.xlabel('Fatalities')
plt.ylabel('Injured')
plt.title(f'Birch Clustering (Threshold={threshold}, Silhouette Score={silhouette_score_value:.2f})')
plt.colorbar(label='Cluster')
st.pyplot()

# Display silhouette score
st.write(f'Silhouette Score: {silhouette_score_value:.2f}')
