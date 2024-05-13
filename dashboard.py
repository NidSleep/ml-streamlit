import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, Birch, AffinityPropagation
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
import mplcursors
from fcmeans import FCM
from io import BytesIO

st.title("Mass Shooting Case's Casualty Visualization")
st.write("""
### Adjust the parameters below to see how clustering changes with different algorithms, and export the results.
""")

# Model selection
model_option = st.selectbox(
    'Choose a clustering model',
    ('K-Means', 't-SNE', 'Fuzzy C-means', 'Birch', 'Affinity Propagation'),
    key='model_selection'
)

# Conditional parameters based on model
n_clusters = 3
if model_option in ['K-Means', 'Fuzzy C-means']:
    n_clusters = st.slider("Select the number of clusters:", min_value=2, max_value=10, value=3, key='n_clusters_slider')

# Load data
@st.cache
def load_data(url):
    df = pd.read_csv(url)
    df.dropna(subset=['Longitude', 'Latitude', 'Fatalities', 'Injured', 'Total victims', 'Policeman Killed'], inplace=True)
    return df

url = 'https://raw.githubusercontent.com/NidSleep/streamlit-example/master/dataset_cleansed.csv'
df = load_data(url)
df_casualty = df[['Fatalities', 'Injured', 'Total victims', 'Policeman Killed', 'S#']]

# Clustering or dimensionality reduction
def perform_model(data, model_option, n_clusters):
    if model_option == 'K-Means':
        model = KMeans(n_clusters=n_clusters)
        labels = model.fit_predict(data)
    elif model_option == 't-SNE':
        tsne = TSNE(n_components=2, random_state=0)
        transformed_data = tsne.fit_transform(data)
        st.pyplot(plt.scatter(transformed_data[:, 0], transformed_data[:, 1], alpha=0.5))
        return None  # No clustering, just visualization
    elif model_option == 'Fuzzy C-means':
        model = FCM(n_clusters=n_clusters)
        model.fit(np.array(data))
        labels = model.u.argmax(axis=1)
    elif model_option == 'Birch':
        model = Birch(n_clusters=None)
        labels = model.fit_predict(data)
    elif model_option == 'Affinity Propagation':
        model = AffinityPropagation(random_state=0)
        labels = model.fit_predict(data)
    return labels

labels = perform_model(df_casualty.iloc[:, :-1], model_option, n_clusters) if model_option != 't-SNE' else None

# Update DataFrame with clusters
if labels is not None:
    df_casualty['Cluster'] = labels
    # Allow users to download the updated DataFrame
    def to_excel(df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
            writer.save()
        processed_data = output.getvalue()
        return processed_data

    st.download_button(label='📥 Download Excel',
                       data=to_excel(df_casualty),
                       file_name='clustered_data.xlsx',
                       mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

# Display DataFrame with clusters in UI
st.dataframe(df_casualty)
