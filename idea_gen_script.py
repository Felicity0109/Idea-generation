import streamlit as st
import pandas as pd
import numpy as np
import re

# text
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# dim reduction & clustering
import umap
import hdbscan

# plotting
import plotly.express as px
import networkx as nx

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
SIMILARITY_THRESHOLD = 0.35

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

@st.cache_data(ttl=3600)
def reduce_embeddings(embeddings, n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST, n_components=UMAP_N_COMPONENTS):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=42)
    return reducer.fit_transform(embeddings)

@st.cache_data(ttl=3600)
def cluster_embeddings(embeddings, min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean', cluster_selection_method='eom')
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

# --- Streamlit App ---
def run_app():
    st.set_page_config(layout='wide', page_title='R&T Idea Mining')
    st.title('R&T Idea Mining')

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

    # minimal validation
    required_cols = ['Idea', 'Research group']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f'Missing required columns: {missing}')
        st.stop()

    if 'preprocessed' not in st.session_state:
        st.session_state['preprocessed'] = False

    if not st.session_state['preprocessed']:
        with st.spinner('Processing ideas...'):
            df['clean_text'] = df['Idea'].astype(str).apply(clean_text)
            texts = df['clean_text'].tolist()
            embeddings = compute_embeddings(texts)
            emb2d = reduce_embeddings(embeddings)
            labels = cluster_embeddings(embeddings)
            df['cluster'] = labels
            df['umap_x'] = emb2d[:,0] if emb2d.shape[1]>=2 else 0.0
            df['umap_y'] = emb2d[:,1] if emb2d.shape[1]>=2 else 0.0
            cluster_terms = extract_top_terms_per_cluster(df['clean_text'].tolist(), df['cluster'].tolist(), top_n=8)
            G = build_similarity_graph(df['Idea'].tolist(), embeddings, threshold=SIMILARITY_THRESHOLD)
            st.session_state.update({
                'df': df,
                'embeddings': embeddings,
                'G': G,
                'cluster_terms': cluster_terms,
                'preprocessed': True
            })

    df = st.session_state['df']

    st.header('Overview')
    c1, c2 = st.columns([2,1])
    with c1:
        st.subheader('UMAP projection of ideas')
        fig = px.scatter(df, x='umap_x', y='umap_y', color=df['cluster'].astype(str),
                         hover_data=['Idea','Research group'],
                         title='UMAP: ideas colored by cluster')
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader('Cluster summary')
        cluster_summary = df.groupby('cluster').agg(
            count=('Idea','count'),
            groups=('Research group', lambda x: ', '.join(sorted(set(x))))
        ).reset_index()
        cluster_summary['top_terms'] = cluster_summary['cluster'].apply(
            lambda r: st.session_state['cluster_terms'].get(f'cluster_{r}', st.session_state['cluster_terms'].get('noise', []))
        )
        st.dataframe(cluster_summary)

    st.markdown('---')
    st.header('Network view (similar ideas)')
    G = st.session_state['G']
    if G.number_of_nodes() == 0:
        st.info('No nodes found for the similarity graph.')
    else:
        pos = nx.spring_layout(G, k=0.5, seed=42)
        node_x, node_y, node_text, node_color = [], [], [], []
        for n in G.nodes():
            x,y = pos.get(n,(0.0,0.0))
            node_x.append(x)
            node_y.append(y)
            row = df.iloc[n]
            hover_preview = (row['Idea'][:200]+'...') if len(str(row['Idea']))>200 else row['Idea']
            node_text.append(f"{row['Idea']} - {row['Research group']}\n{hover_preview}")
            node_color.append(str(row['cluster']))
        net_fig = px.scatter(x=node_x, y=node_y, color=node_color, hover_name=node_text,
                             labels={'color':'cluster'}, title='Similarity network')
        st.plotly_chart(net_fig, use_container_width=True)

    st.markdown('---')
    st.header('Cluster drilldown')
    sel_cluster = st.selectbox('Select cluster', options=sorted(df['cluster'].unique().tolist()))
    sub = df[df['cluster']==sel_cluster]
    st.subheader(f'Cluster {sel_cluster} — {len(sub)} ideas')
    st.write('Top terms:', st.session_state['cluster_terms'].get(f'cluster_{sel_cluster}', st.session_state['cluster_terms'].get('noise', [])))
    st.dataframe(sub[['Idea','Research group']])

    
    st.markdown('---')
    st.header('Word Cloud of All Ideas')

    # Combine all clean text
    all_text = " ".join(df['clean_text'].tolist())

    if all_text.strip():  # check that there is text
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
    tmp = df[['Idea','Research group']].copy()
    csv = tmp.to_csv(index=False).encode('utf-8')
    st.download_button('Download processed CSV', data=csv, file_name='ideas_processed.csv', mime='text/csv')


if __name__ == '__main__':
    run_app()
