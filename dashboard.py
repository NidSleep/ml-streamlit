import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, Birch, AffinityPropagation
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
import mplcursors
from fcmeans import FCM

st.title("Mass Shooting Case's Casualty Visualization")
st.write("""
### Adjust the parameters below to see how clustering changes with different algorithms.
""")

# Model selection
model_option = st.selectbox(
    'Choose a clustering model',
    ('K-Means', 't-SNE + K-Means', 'Fuzzy C-means', 'Birch', 'Affinity Propagation'),
    key='model_selection'
)

# Load data
@st.cache
def load_data(url):
    df = pd.read_csv(url)
    df.dropna(subset=['Longitude', 'Latitude', 'Fatalities', 'Injured', 'Total victims', 'Policeman Killed'], inplace=True)
    return df

url = 'https://raw.githubusercontent.com/NidSleep/streamlit-example/master/dataset_cleansed.csv'
df = load_data(url)
df_casualty = df[['Fatalities', 'Injured', 'Total victims', 'Policeman Killed', 'S#']]

# Clustering
def perform_clustering(data, model_option):
    if model_option == 'K-Means':
        model = KMeans(n_clusters=3)
    elif model_option == 't-SNE + K-Means':
        tsne = TSNE(n_components=2)
        transformed_data = tsne.fit_transform(data)
        model = KMeans(n_clusters=3)
        return model.fit_predict(transformed_data), model
    elif model_option == 'Fuzzy C-means':
        model = FCM(n_clusters=3)
        model.fit(np.array(data))
        return model.u.argmax(axis=1), model
    elif model_option == 'Birch':
        model = Birch(n_clusters=None)
    elif model_option == 'Affinity Propagation':
        model = AffinityPropagation(random_state=0)
    return model.fit_predict(data), model

cluster_labels, model = perform_clustering(df_casualty.iloc[:, :-1], model_option)

# Calculate metrics
if model_option == 'Fuzzy C-means':
    silhouette_score_value = 'N/A for Fuzzy C-means'
else:
    silhouette_score_value = silhouette_score(df_casualty.iloc[:, :-1], cluster_labels)

davies_bouldin_score_value = davies_bouldin_score(df_casualty.iloc[:, :-1], cluster_labels)

# Plot
fig, ax = plt.subplots()
scatter = ax.scatter(df_casualty['Fatalities'], df_casualty['Injured'], c=cluster_labels, cmap='viridis', alpha=0.5)
colorbar = plt.colorbar(scatter, ax=ax, label='Cluster')

# mplcursors for hover info
mplcursors.cursor(scatter).connect(
    "add",
    lambda sel: sel.annotation.set_text(f'ID: {df_casualty.iloc[sel.target.index]["S#"]}')
)
plt.xlabel('Fatalities')
plt.ylabel('Injured')
plt.title(f'Clustering with {model_option}')

st.pyplot(fig)

# Display metrics
if silhouette_score_value != 'N/A for Fuzzy C-means':
    st.write(f'Silhouette Score: {silhouette_score_value:.2f}')
else:
    st.write(f'Silhouette Score: {silhouette_score_value}')

st.write(f'Davies-Bouldin Score: {davies_bouldin_score_value:.2f}')
