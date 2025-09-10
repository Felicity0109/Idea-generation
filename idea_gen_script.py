import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.filterwarnings("ignore")  

# text
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# dim reduction & clustering
import umap
from sklearn.decomposition import PCA
import hdbscan

# plotting
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# caching
from functools import lru_cache

# --- stopwords using sklearn to avoid nltk dependency ---
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
STOPWORDS = set(ENGLISH_STOP_WORDS)

# --- CONFIG ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_N_COMPONENTS = 2
HDBSCAN_MIN_CLUSTER_SIZE = 3
SIMILARITY_THRESHOLD = 0.20

# --- utilities ---
def clean_text(text):
    if not isinstance(text, str):
        return ""
    txt = text.lower()
    txt = re.sub(r"https?://\S+|www\.\S+", " ", txt)
    txt = re.sub(r"[^a-z0-9\s]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    tokens = [t for t in txt.split() if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)

@st.cache_resource
def load_embedding_model(name=EMBEDDING_MODEL):
    return SentenceTransformer(name)

@st.cache_data(ttl=3600)
def compute_embeddings(texts, model_name=EMBEDDING_MODEL):
    model = load_embedding_model(model_name)
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings

def reduce_embeddings_dynamic(embeddings, n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST, n_components=UMAP_N_COMPONENTS):
    n_rows = embeddings.shape[0]
    if n_rows <= 50:
        reducer = PCA(n_components=n_components, random_state=42)
        reduced = reducer.fit_transform(embeddings)
        method = 'PCA'
    else:
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=42)
        reduced = reducer.fit_transform(embeddings)
        method = 'UMAP'
    return reduced, method

@st.cache_data(ttl=3600)
def cluster_embeddings(embeddings, min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean', cluster_selection_method='eom', cluster_selection_epsilon=0.1)
    return clusterer.fit_predict(embeddings)

@st.cache_data(ttl=3600)
def extract_top_terms_per_cluster(docs, labels, top_n=8):
    df = pd.DataFrame({'doc': docs, 'label': labels})
    docs_by_label = df.groupby('label')['doc'].apply(lambda x: " ".join(x)).to_dict()
    results = {}
    vect = TfidfVectorizer(max_features=2000, ngram_range=(1,2))
    for label, bigdoc in docs_by_label.items():
        label_name = 'noise' if label == -1 else f'cluster_{label}'
        X = vect.fit_transform([bigdoc])
        terms = np.array(vect.get_feature_names_out())
        scores = X.toarray()[0]
        top_idx = scores.argsort()[::-1][:top_n]
        results[label_name] = list(terms[top_idx]) if len(terms) > 0 else []
    return results

def build_similarity_graph(docs, embeddings, threshold=SIMILARITY_THRESHOLD):
    if len(docs) == 0:
        return nx.Graph()
    sim = cosine_similarity(embeddings)
    G = nx.Graph()
    for i, doc in enumerate(docs):
        G.add_node(i, label=doc)
    n = len(docs)
    for i in range(n):
        for j in range(i+1, n):
            s = float(sim[i,j])
            if np.isfinite(s) and s >= threshold:
                G.add_edge(i,j,weight=s)
    return G

# --- Topic Modeling using simple TF-IDF + SVD ---
@st.cache_data(ttl=3600)
def topic_modeling(docs, n_topics=5):
    vect = CountVectorizer(max_features=2000, stop_words=STOPWORDS)
    X = vect.fit_transform(docs)
    svd = TruncatedSVD(n_components=n_topics, random_state=42)
    topics = svd.fit_transform(X)
    topic_terms = {}
    terms = np.array(vect.get_feature_names_out())
    for i, comp in enumerate(svd.components_):
        top_idx = comp.argsort()[::-1][:8]
        topic_terms[f'topic_{i}'] = list(terms[top_idx])
    return topics, topic_terms

# --- Compute novelty score based on embedding distance ---
@st.cache_data(ttl=3600)
def compute_novelty(embeddings):
    # novelty = mean distance to all other ideas
    dists = euclidean_distances(embeddings)
    np.fill_diagonal(dists, np.nan)  # ignore self-distance
    novelty = np.nanmean(dists, axis=1)
    # normalize between 0-1
    novelty = MinMaxScaler().fit_transform(novelty.reshape(-1,1)).flatten()
    return novelty

# --- Streamlit App ---
def run_app():
    st.set_page_config(layout='wide', page_title='Sasol R&T Idea Mining')
    st.title('Sasol R&T Idea Mining')

    # Footer
    st.markdown(
        """
        ---
        © 2025 R&T Idea Mining | Developed by Sasol Research and Technology - Fundamental Science Research
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.header('Data source')
    uploaded_file = st.sidebar.file_uploader('Upload ideas Excel (.xlsx)', type=['xlsx'])
    if uploaded_file is None:
        st.info('Upload a .xlsx file with columns: Idea, Research group')
        st.stop()

    df = pd.read_excel(uploaded_file)
    # Make column names lowercase
    df.columns = [c.lower() for c in df.columns]

    # minimal validation
    required_cols = ['idea', 'research group']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f'Missing required columns: {missing}')
        st.stop()

    if 'preprocessed' not in st.session_state:
        st.session_state['preprocessed'] = False

    if not st.session_state['preprocessed']:
        with st.spinner('Processing ideas...'):
            df['clean_text'] = df['idea'].astype(str).apply(clean_text)
            texts = df['clean_text'].tolist()
            embeddings = compute_embeddings(texts)
            emb2d, method = reduce_embeddings_dynamic(embeddings)
            labels = cluster_embeddings(embeddings)
            df['cluster'] = labels
            df['umap_x'] = emb2d[:,0] if emb2d.shape[1]>=2 else 0.0
            df['umap_y'] = emb2d[:,1] if emb2d.shape[1]>=2 else 0.0
            cluster_terms = extract_top_terms_per_cluster(df['clean_text'].tolist(), df['cluster'].tolist(), top_n=8)
            G = build_similarity_graph(df['idea'].tolist(), embeddings, threshold=SIMILARITY_THRESHOLD)
            topics_matrix, topic_terms = topic_modeling(df['clean_text'].tolist(), n_topics=5)
            df['topic'] = topics_matrix.argmax(axis=1)  # assign dominant topic per idea
            df['novelty'] = compute_novelty(embeddings)
            
            st.session_state.update({
                'df': df,
                'embeddings': embeddings,
                'G': G,
                'cluster_terms': cluster_terms,
                'method': method,
                'topic_terms': topic_terms,
                'preprocessed': True
            })

    # --- Filtering UI ---
    st.sidebar.header("Filters")
    cluster_options = sorted(df['cluster'].unique())
    group_options = sorted(df['research group'].unique())
    topic_options = sorted(df['topic'].unique())

    sel_cluster = st.sidebar.multiselect("Cluster", options=cluster_options, default=cluster_options)
    sel_group = st.sidebar.multiselect("Research group", options=group_options, default=group_options)
    sel_topic = st.sidebar.multiselect("Topic", options=topic_options, default=topic_options)
    novelty_min, novelty_max = st.sidebar.slider("Novelty score", min_value=0.0, max_value=1.0, value=(0.0,1.0), step=0.01)

    filtered_df = df[
        (df['cluster'].isin(sel_cluster)) &
        (df['research group'].isin(sel_group)) &
        (df['topic'].isin(sel_topic)) &
        (df['novelty'].between(novelty_min, novelty_max))
    ]

    st.header(f"Filtered Ideas ({len(filtered_df)})")
    st.dataframe(filtered_df[['idea','research group','cluster','topic','novelty']])

    df = st.session_state['df']
    method = st.session_state['method']

    st.header('Overview')
    st.write(f"Dimensionality reduction method used: {method}")
    c1, c2 = st.columns([2,1])
    with c1:
        st.subheader(f'{method} projection of ideas')
        fig = px.scatter(df, x='umap_x', y='umap_y', color=df['cluster'].astype(str),
                         hover_data=['idea','research group'],
                         title=f'{method}: ideas colored by cluster')
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader('Cluster summary')
        cluster_summary = df.groupby('cluster').agg(
            count=('idea','count'),
            groups=('research group', lambda x: ', '.join(sorted(set(x))))
        ).reset_index()
        cluster_summary['top_terms'] = cluster_summary['cluster'].apply(
            lambda r: st.session_state['cluster_terms'].get(f'cluster_{r}', st.session_state['cluster_terms'].get('noise', []))
        )
        st.dataframe(cluster_summary)

    st.markdown('---')
    st.header('Network view (similar ideas)')
    G_full = st.session_state['G']
    
    if G_full.number_of_nodes() == 0:
        st.info('No nodes found for the similarity graph.')
    else:
        zoom_top_n = st.checkbox('Focus on top 20 most similar ideas', value=False)
        if zoom_top_n:
            sim = cosine_similarity(st.session_state['embeddings'])
            n = len(sim)
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    edges.append((i, j, sim[i,j]))

            edges_sorted = sorted(edges, key=lambda x: x[2], reverse=True)
            top_edges = edges_sorted[:20]
            # Build subgraph
            G = nx.Graph()
            for i in range(n):
                G.add_node(i, label=st.session_state['df'].iloc[i]['idea'])
            for i, j, s in top_edges:
                G.add_edge(i, j, weight=s)

        else:
            G = G_full
                
        pos = nx.spring_layout(G, k=0.5, seed=42)
        # Edges
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x += [x0, x1, None]  # None separates line segments
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        # Nodes
        node_x = []
        node_y = []
        node_color = []
        node_text = []

        for n in G.nodes():
            x, y = pos[n]
            node_x.append(x)
            node_y.append(y)
            row = st.session_state['df'].iloc[n]

    # Map cluster to numeric color (noise=-1 becomes 0)
            cluster_val = 0 if row['cluster'] == -1 else row['cluster']
            node_color.append(cluster_val)

            hover_preview = (row['idea'][:200]+'...') if len(str(row['idea']))>200 else row['idea']
            node_text.append(f"{row['idea']} - {row['research group']}\n{hover_preview}")

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            marker=dict(
                size=15,
                color=node_color,
                colorscale='Viridis',  # numeric values now compatible
                line=dict(width=2, color='black'),
                showscale=True,
                colorbar=dict(title='Cluster')
            ),
            text=node_text,
            hoverinfo='text'
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Similarity Network',
                        showlegend=False,
                        hovermode='closest',
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))

        st.plotly_chart(fig, use_container_width=True)

    st.markdown('---')
    st.header('Cluster drilldown')
    sel_cluster = st.selectbox('Select cluster', options=sorted(df['cluster'].unique().tolist()))
    sub = df[df['cluster']==sel_cluster]
    st.subheader(f'Cluster {sel_cluster} — {len(sub)} ideas')
    st.write('Top terms:', st.session_state['cluster_terms'].get(f'cluster_{sel_cluster}', st.session_state['cluster_terms'].get('noise', [])))
    st.dataframe(sub[['idea','research group']])

    st.markdown('---')
    st.header('Word Cloud of All Ideas')
    all_text = " ".join(df['clean_text'].tolist())
    if all_text.strip():
        wc = WordCloud(width=800, height=400, background_color='white',
                       colormap='viridis', stopwords=STOPWORDS).generate(all_text)
        fig, ax = plt.subplots(figsize=(10,5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.info("No text available for word cloud generation.")

    st.markdown('---')
    st.header('Export & Utilities')
    tmp = df[['idea','research group']].copy()
    csv = tmp.to_csv(index=False).encode('utf-8')
    st.download_button('Download processed CSV', data=csv, file_name='ideas_processed.csv', mime='text/csv')


if __name__ == '__main__':
    run_app()
