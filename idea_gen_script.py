import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
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
from sklearn.cluster import KMeans

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
UMAP_N_NEIGHBORS = 5
UMAP_MIN_DIST = 0.01
UMAP_N_COMPONENTS = 2
HDBSCAN_MIN_CLUSTER_SIZE = 5
SIMILARITY_THRESHOLD = 0.8

# --- Utilities ---
def clean_text(text):
    if not isinstance(text, str):
        return ""
    txt = text.lower()
    txt = re.sub(r"https?://\S+|www\.\S+", " ", txt)
    txt = re.sub(r"[^a-z0-9\s]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    tokens = [t for t in txt.split() 
        if t not in STOPWORDS and t != "sasol" and len(t) > 1]
    return " ".join(tokens)

@st.cache_resource
def load_embedding_model(name=EMBEDDING_MODEL):
    from sentence_transformers import SentenceTransformer
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
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, 
                                metric='euclidean',
                                cluster_selection_method='eom',
                                cluster_selection_epsilon=0.1)
    labels = clusterer.fit_predict(embeddings)
    n_noise = np.sum(labels == -1)
    if n_noise > len(labels) * 0.7:  # if >70% points are -1, fallback
        n_clusters = min(5, len(embeddings)//2)  # up to 5 clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        print(f"⚠️ HDBSCAN failed, using KMeans with {n_clusters} clusters")    
    return labels
    
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

def build_similarity_graph(embeddings, top_k=5, min_sim=0.6):
    """
    Build similarity graph with hybrid logic:
    - keep top_k most similar neighbors per node
    - only add edges if similarity >= min_sim
    """
    sim_matrix = cosine_similarity(embeddings)
    n = len(embeddings)
    G = nx.Graph()
    for i in range(n):
        sims = [(j, sim_matrix[i, j]) for j in range(n) if i != j]
        sims = sorted(sims, key=lambda x: x[1], reverse=True)[:top_k]
        for j, sim in sims:
            if sim >= min_sim:
                G.add_edge(i, j, weight=sim)
    return G

def subgraph_for_cluster(G, cluster_labels, cluster_id):
    """
    Extract subgraph for nodes belonging to a specific cluster.
    (Keeps node indices aligned with the global DataFrame.)
    """
    nodes_in_cluster = [i for i, c in enumerate(cluster_labels) if c == cluster_id]
    return G.subgraph(nodes_in_cluster).copy()

@st.cache_data(ttl=3600)
def topic_modeling(docs, n_topics=5):
    vect = CountVectorizer(max_features=3000, stop_words=list(STOPWORDS))
    X = vect.fit_transform(docs)
    svd = TruncatedSVD(n_components=n_topics, random_state=42)
    topics = svd.fit_transform(X)
    topic_terms = {}
    terms = np.array(vect.get_feature_names_out())
    for i, comp in enumerate(svd.components_):
        top_idx = comp.argsort()[::-1][:8]
        topic_terms[f'topic_{i}'] = list(terms[top_idx])
    return topics, topic_terms

@st.cache_data(ttl=3600)
def compute_novelty(embeddings):
    dists = euclidean_distances(embeddings)
    np.fill_diagonal(dists, np.nan)
    novelty = np.nanmean(dists, axis=1)
    novelty = MinMaxScaler().fit_transform(novelty.reshape(-1,1)).flatten()
    return novelty

# --- Network plotting utility ---
def plot_network(G, subset_df=None, title='Similarity Network'):
    if G.number_of_nodes() == 0:
        st.info('No nodes found for the similarity graph.')
        return

    pos = nx.spring_layout(G, k=0.5, seed=42)
    
    edge_x, edge_y = [], []
    
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#888'),
                            hoverinfo='none', mode='lines')
    df_plot = subset_df if subset_df is not None else st.session_state['df']
    displayed_indices = [n for n in G.nodes() if n in df_plot.index]
    clusters = sorted(df_plot.loc[displayed_indices, 'cluster'].unique())
    cluster_to_color = {c: i for i, c in enumerate(clusters)}

    node_x, node_y, node_color, node_text = [], [], [], []

    clusters = sorted(df_plot['cluster'].unique())
    cluster_to_color = {c: i for i, c in enumerate(clusters)}

    for node in displayed_indices:
        row = df_plot.loc[node]
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        cluster_val = cluster_to_color.get(row['cluster'], 0)
        node_color.append(cluster_val)
        hover_preview = (row['idea'][:200] + '...') if len(str(row['idea'])) > 200 else row['idea']
        node_text.append(f"{row['idea']} - {row['research group']}\n{hover_preview}")

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers',
        marker=dict(
            size=15,
            color=node_color,
            colorscale='Viridis',
            line=dict(width=2, color='black'),
            showscale=True,
            colorbar=dict(title='Cluster', tickvals=list(cluster_to_color.values()),
                          ticktext=[str(c) for c in clusters])
        ),
        text=node_text,
        hoverinfo='text'
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=title,
                        showlegend=False,
                        hovermode='closest',
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))
    st.plotly_chart(fig, use_container_width=True)

# --- Streamlit App ---
def run_app():
    st.set_page_config(layout='wide', page_title='Sasol R&T Idea Mining')
    st.title('Sasol R&T Idea Exploration Dashboard')
    st.markdown('---')
    st.header("How the it Works")

    st.markdown("""
    **Step-by-step workflow of idea analysis:**

    :open_file_folder: **1. Upload Ideas**  
    Upload your ideas in an Excel (.xlsx) file with columns `Idea` and `Research group`.

    :gear: **2. Text Preprocessing**  
    Ideas are cleaned, lowercased, and stopwords removed to prepare for analysis.

    :bar_chart: **3. Embedding & Clustering**  
    Each idea is converted into a numerical vector (embedding).  
    HDBSCAN or KMeans groups similar ideas into clusters.

    :mag_right: **4. Idea Similarity Network**  
    Shows how ideas are semantically connected.  
    Options:  
    - Top 20 most similar ideas  
    - Network per cluster

    :cloud: **5. Cluster-level Word Clouds & Frequency Plots**  
    Highlights main terms and their frequencies within each cluster.

    :bulb: **6. Topic Modeling**  
    Identifies overarching **“idea themes”** for easier interpretation.

    :star2: **7. Novelty Score**  
    Highlights unique or unusual ideas for attention.

    **Tip:** Explore clusters, networks, and word clouds together — networks show relationships, clusters summarize themes, and novelty scores highlight standout ideas.
    """)

    
    st.markdown("""
        ---
        © 2025 R&T Idea Mining | Developed by Sasol Research and Technology - Fundamental Science Research
        """, unsafe_allow_html=True)

    # --- Upload data ---
    st.sidebar.header('Data source')
    uploaded_file = st.sidebar.file_uploader('Upload ideas Excel (.xlsx)', type=['xlsx'])
    if uploaded_file is None:
        st.info('Upload a .xlsx file with columns: Idea, Research group')
        st.stop()

    df = pd.read_excel(uploaded_file)
    df.columns = [c.lower() for c in df.columns]
    required_cols = ['idea', 'research group']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f'Missing required columns: {missing}')
        st.stop()

    # --- Preprocessing ---
    if 'preprocessed' not in st.session_state or not st.session_state['preprocessed']:
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
            G = build_similarity_graph(embeddings, top_k=5, min_sim=0.6)  # tune top_k/min_sim as needed
            topics_matrix, topic_terms = topic_modeling(df['clean_text'].tolist(), n_topics=5)
            df['topic'] = topics_matrix.argmax(axis=1)
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
    relabelled_topics = {f'Idea Theme {i+1}': words 
                     for i, (topic, words) in enumerate(st.session_state['topic_terms'].items())}
    st.session_state['topic_terms'] = relabelled_topics
    
    df = st.session_state['df']
    method = st.session_state['method']

    # --- Filters ---
    st.sidebar.header("Filters")
    cluster_options = sorted(df['cluster'].unique())
    group_options = sorted(df['research group'].unique())
    topic_options = sorted(df['topic'].unique())

    sel_cluster_filter = st.sidebar.multiselect("Cluster", options=cluster_options, default=cluster_options)
    sel_group = st.sidebar.multiselect("Research group", options=group_options, default=group_options)
    sel_topic = st.sidebar.multiselect("Topic", options=topic_options, default=topic_options)

    st.sidebar.markdown("""
    **Novelty Score:**  
    Measures how unique an idea is compared to all others, with higher values indicating more distinctive ideas.
    """)
    novelty_min, novelty_max = st.sidebar.slider("Novelty score", 0.0, 1.0, (0.0,1.0), 0.01)

    filtered_df = df[
        (df['cluster'].isin(sel_cluster_filter)) &
        (df['research group'].isin(sel_group)) &
        (df['topic'].isin(sel_topic)) &
        (df['novelty'].between(novelty_min, novelty_max))
    ]

    st.header(f"Filtered Ideas ({len(filtered_df)})")
    st.dataframe(filtered_df[['idea','research group','cluster','topic','novelty']])

    # --- Top Words per Topic ---
    st.markdown('---')
    st.header('Top Words per Idea Theme')
    for theme, words in st.session_state['topic_terms'].items():
        st.subheader(f"{theme}")
        st.write(", ".join(words))

    # --- Overview ---
    st.markdown('---')
    st.header('Overview')
    st.write(f"Dimensionality reduction method used: {method}")
    c1, c2 = st.columns([2,1])
    
    with st.expander("Click to view idea projection (UMAP/PCA) and cluster summary"):
        c1, c2 = st.columns([2,1])
    
    with c1:
        st.subheader(f'{method} projection of ideas')
        fig = px.scatter(
            df, 
            x='umap_x', y='umap_y', 
            color=df['cluster'].astype(str),
            hover_data=['idea','research group'],
            title=f'{method}: ideas colored by cluster'
        )
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

    # --- Similarity Network ---
    st.markdown('---')
    st.header('Network view of simalar ideas across the dataset')
    st.info("Nodes represent ideas, edges indicate strong semantic similarity. "
        "Hover over nodes to see idea text and research group. "
        "Top 20 filter shows the strongest connections.")
    G_full = st.session_state['G']

    if G_full.number_of_nodes() == 0:
        st.info('No nodes found for the similarity graph.')
    else:
        zoom_top_n = st.checkbox('Focus on top 20 most similar ideas', value=False)
        cluster_focus = st.checkbox('Show network by cluster', value=False)

    if zoom_top_n:
        sim = cosine_similarity(st.session_state['embeddings'])
        n = len(sim)
        edges = [(i, j, sim[i,j]) for i in range(n) for j in range(i+1, n)]
        top_edges = sorted(edges, key=lambda x: x[2], reverse=True)[:20]
        
        G = nx.Graph()
        for i in range(n):
            G.add_node(i, label=st.session_state['df'].iloc[i]['idea'])
        for i, j, s in top_edges:
            G.add_edge(i, j, weight=s)
        plot_network(G)
        
    elif cluster_focus:
        clusters = sorted(df['cluster'].unique())
        sel_cluster_net = st.selectbox('Select cluster to display', options=clusters)
        subset_df = st.session_state['df'][st.session_state['df']['cluster'] == sel_cluster_net]

        if not subset_df.empty:
            sim = cosine_similarity(st.session_state['embeddings'][subset_df.index])
            n = len(sim)
            G_cluster = nx.Graph()
            for i in range(n):
                G_cluster.add_node(i, label=subset_df.iloc[i]['idea'])
            for i in range(n):
                for j in range(i+1, n):
                    s = sim[i,j]
                    if np.isfinite(s) and s >= SIMILARITY_THRESHOLD:
                        G_cluster.add_edge(i, j, weight=s)
            plot_network(G_cluster, subset_df=subset_df)
    else:
        plot_network(G_full)

    # --- Cluster drilldown + word cloud ---
    st.markdown('---')
    st.header('Cluster drilldown')

    sel_cluster_wc = st.selectbox('Select cluster', options=sorted(df['cluster'].unique()), key='cluster_wc')
    sub = df[df['cluster'] == sel_cluster_wc]
    st.subheader(f'Cluster {sel_cluster_wc} — {len(sub)} ideas')
    st.dataframe(sub[['idea', 'research group']])

# --- Word Network + Frequency Plot ---
    st.subheader("Word Network & Frequency plots")

    if not sub.empty:
    # prepare text
        text = " ".join(sub['idea'].astype(str).tolist())
        text_clean = re.sub(r"[^a-z0-9\s]", " ", text.lower())
        words = [w for w in text_clean.split() if w not in STOPWORDS and len(w) > 1]

    # frequency distribution
        freq_dist = Counter(words)
        top_words = [w for w, _ in freq_dist.most_common(30)]

    # TF-IDF + cosine similarity for word network
        vecs = TfidfVectorizer(vocabulary=top_words, stop_words=list(STOPWORDS)).fit_transform(sub['idea'])
        cos_sim = cosine_similarity(vecs.T)

        G = nx.Graph()
        for i, w1 in enumerate(top_words):
            for j, w2 in enumerate(top_words):
                if i < j and cos_sim[i, j] > 0.1:  # edge threshold
                    G.add_edge(w1, w2, weight=cos_sim[i, j])

    # plot network
        fig, ax = plt.subplots(figsize=(8, 6))
        pos = nx.spring_layout(G, k=0.5, seed=42)
        nx.draw(G, pos, with_labels=True, node_size=500, font_size=10, ax=ax)
        st.pyplot(fig)

    # frequency plot
        freq_df = pd.DataFrame(freq_dist.most_common(20), columns=["Word", "Frequency"])
        st.bar_chart(freq_df.set_index("Word"))
        
    # --- Cluster-level Word Cloud ---
        st.subheader("Word Cloud (Cluster)")
        wc = WordCloud(
            width=800, height=400,
            background_color='white',
            colormap='viridis',
            stopwords=STOPWORDS
        ).generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

# --- Word Cloud for all ideas ---
    st.markdown('---')
    st.header('Word Cloud of All Ideas')
    all_text = " ".join(df['clean_text'].astype(str).tolist())
    if all_text.strip():
        wc = WordCloud(width=800, height=400, background_color='white',
                    colormap='viridis', stopwords=STOPWORDS).generate(all_text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.info("No text available for word cloud generation.")
    
# --- Export ---
    st.markdown('---')
    st.header('Export')
    st.write("Download ideas as CSV file. Select filters on the sidebar (Cluster, Research group, Topic, Novelty) to narrow down the ideas before downloading.")

    if not filtered_df.empty:
        columns_to_export = ['idea','research group','cluster','topic','novelty','clean_text']
        csv_filtered = filtered_df[columns_to_export].to_csv(index=False).encode('utf-8')
        st.download_button(
            label='Download filtered CSV',
            data=csv_filtered,
            file_name='ideas_filtered.csv',
            mime='text/csv'
        )
    else:
        st.info("No ideas match the current filters to download.")

if __name__ == '__main__':
    run_app()
