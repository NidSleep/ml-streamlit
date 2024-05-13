import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, Birch, AffinityPropagation
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
from fcmeans import FCM

st.title("Mass Shooting Case's Casualty Visualization")
st.write("""
### Adjust the parameters below to see how clustering changes with different algorithms.
""")

# Model selection
model_option = st.selectbox(
    'Choose a visualization or clustering model',
    ('K-Means', 't-SNE', 'Fuzzy C-means', 'Birch', 'Affinity Propagation'),
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

# Display interactive table
st.dataframe(df_casualty)

# Model-specific parameters
if model_option in ['K-Means', 'Fuzzy C-means', 'Birch']:
    n_clusters = st.slider("Number of clusters", 2, 10, 3, key='n_clusters')
else:
    n_clusters = None
random_state = st.number_input("Random state (seed)", min_value=0, max_value=100, value=42, key='random_state')

# t-SNE specific parameters
if model_option == 't-SNE':
    n_components = st.slider("Number of components for t-SNE", 2, 3, 2, key='n_components')
    perplexity = st.slider("Perplexity for t-SNE", 5, 50, 30, key='perplexity')
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    transformed_data = tsne.fit_transform(df_casualty.iloc[:, :-1])
    fig, ax = plt.subplots()
    ax.scatter(transformed_data[:, 0], transformed_data[:, 1], alpha=0.5)
    plt.title('t-SNE Visualization')
    st.pyplot(fig)
else:
    # Clustering function call and operations
    labels = perform_model(df_casualty.iloc[:, :-1], model_option, n_clusters, random_state)
    if labels is not None:
        df_casualty['Cluster'] = labels  # Add cluster labels to DataFrame
        fig, ax = plt.subplots()
        scatter = ax.scatter(df_casualty['Fatalities'], df_casualty['Injured'], c=labels, cmap='viridis', alpha=0.5)
        colorbar = plt.colorbar(scatter, ax=ax, label='Cluster')
        plt.xlabel('Fatalities')
        plt.ylabel('Injured')
        plt.title(f'Clustering with {model_option}')
        st.pyplot(fig)
        if model_option != 'Fuzzy C-means':
            silhouette_score_value = silhouette_score(df_casualty.iloc[:, :-1], labels)
            davies_bouldin_score_value = davies_bouldin_score(df_casualty.iloc[:, :-1], labels)
            st.write(f'Silhouette Score: {silhouette_score_value:.2f}')
            st.write(f'Davies-Bouldin Score: {davies_bouldin_score_value:.2f}')
        # Download functionality for the updated DataFrame
        csv = df_casualty.to_csv(index=False).encode('utf-8')
        st.download_button("Download data as CSV", csv, "clustered_data.csv", "text/csv", key='download-csv')

# Function to perform model operations
def perform_model(data, model_option, n_clusters, random_state):
    if model_option == 'K-Means':
        model = KMeans(n_clusters=n_clusters, random_state=random_state)
        labels = model.fit_predict(data)
    elif model_option == 't-SNE':
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
        transformed_data = tsne.fit_transform(data)
        fig, ax = plt.subplots()
        ax.scatter(transformed_data[:, 0], transformed_data[:, 1], alpha=0.5)
        plt.title('t-SNE Visualization')
        st.pyplot(fig)
        return None
    elif model_option == 'Fuzzy C-means':
        model = FCM(n_clusters=n_clusters, random_state=random_state)
        model.fit(np.array(data))
        labels = model.u.argmax(axis=1)
    elif model_option == 'Birch':
        model = Birch(n_clusters=n_clusters if n_clusters else None)
        labels = model.fit_predict(data)
    elif model_option == 'Affinity Propagation':
        model = AffinityPropagation(random_state=random_state)
        labels = model.fit_predict(data)
    return labels

# Perform clustering or dimensionality reduction
if model_option != 't-SNE':
    labels = perform_model(df_casualty.iloc[:, :-1], model_option, n_clusters, random_state)
    if labels is not None:
        df_casualty['Cluster'] = labels
        # Plot results
        fig, ax = plt.subplots()
        scatter = ax.scatter(df_casualty['Fatalities'], df_casualty['Injured'], c=labels, cmap='viridis', alpha=0.5)
        plt.colorbar(scatter, ax=ax, label='Cluster')
        plt.xlabel('Fatalities')
        plt.ylabel('Injured')
        plt.title(f'Clustering with {model_option}')
        st.pyplot(fig)

    
if labels is not None:
    df_casualty['Cluster'] = labels  # Add cluster labels to the DataFrame
    # Calculate metrics
    silhouette_score_value = 'N/A for Fuzzy C-means' if model_option == 'Fuzzy C-means' else silhouette_score(df_casualty.iloc[:, :-1], labels)
    davies_bouldin_score_value = davies_bouldin_score(df_casualty.iloc[:, :-1], labels)

    # Plot
    fig, ax = plt.subplots()
    scatter = ax.scatter(df_casualty['Fatalities'], df_casualty['Injured'], c=labels, cmap='viridis', alpha=0.5)
    plt.xlabel('Fatalities')
    plt.ylabel('Injured')
    plt.title(f'Clustering with {model_option}')
    # Customize legend as per the example
    legend = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend)
    st.pyplot(fig)

    # Display metrics
    if silhouette_score_value != 'N/A for Fuzzy C-means':
        st.write(f'Silhouette Score: {silhouette_score_value:.2f}')
    st.write(f'Davies-Bouldin Score: {davies_bouldin_score_value:.2f}')

    # Function to convert DataFrame to CSV for download
    @st.cache
    def convert_df_to_csv(df):
        return df.to_csv().encode('utf-8')

    csv = convert_df_to_csv(df_casualty)  # Convert DataFrame to CSV
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='clustered_data.csv',
        mime='text/csv',
    )
