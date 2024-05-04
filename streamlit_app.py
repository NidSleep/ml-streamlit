import streamlit as st
import pandas as pd
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
import mplcursors  # Import mplcursors library

# Title
st.title('Birch Clustering Visualization')

# Sidebar for threshold selection
threshold = st.sidebar.slider('Threshold', min_value=0.1, max_value=0.5, step=0.1, value=0.1)

# Load the data from GitHub raw URL
url = 'https://raw.githubusercontent.com/your_username/your_repository/main/dataset_cleansed.csv'
df = pd.read_csv(url)

# Selecting relevant columns and handling missing values
df_coordinates = df[['Longitude', 'Latitude', 'Fatalities', 'Injured', 'Total victims', 'Policeman Killed', 'Case ID']]
df_coordinates.dropna(inplace=True)
df_casualty = df_coordinates[['Fatalities', 'Injured', 'Total victims', 'Policeman Killed', '#S']]  # Include Case ID

# Calculate silhouette score
birch = Birch(threshold=threshold, n_clusters=None)
cluster_labels = birch.fit_predict(df_casualty.iloc[:, :-1])  # Exclude Case ID for clustering
silhouette_score_value = silhouette_score(df_casualty.iloc[:, :-1], cluster_labels)

# Plot the clustering results
scatter = plt.scatter(df_casualty['Fatalities'], df_casualty['Injured'], c=cluster_labels, cmap='viridis', alpha=0.5)

# Enable mplcursors
mplcursors.cursor(scatter, hover=True).connect('add', lambda sel: sel.annotation.set_text(df_casualty.iloc[sel.target.index][-1]))

plt.xlabel('Fatalities')
plt.ylabel('Injured')
plt.title(f'Birch Clustering (Threshold={threshold}, Silhouette Score={silhouette_score_value:.2f})')
plt.colorbar(label='Cluster')
st.pyplot()

# Display silhouette score
st.write(f'Silhouette Score: {silhouette_score_value:.2f}')
