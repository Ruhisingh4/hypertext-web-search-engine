import numpy as np
import random
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- DATA ----------------
def generate_documents(n=100):
    base_text = "ranking algorithms for large scale web search"
    return [f"{base_text} document {i}" for i in range(n)]

documents = generate_documents()

# ---------------- GRAPH ----------------
def create_web_graph(num_docs=100, avg_links=5):
    G = nx.DiGraph()
    for i in range(num_docs):
        links = random.sample(range(num_docs), avg_links)
        for l in links:
            if l != i:
                G.add_edge(i, l)
    return G

G = create_web_graph()

def compute_pagerank(G):
    return nx.pagerank(G, alpha=0.85)

pagerank_scores = compute_pagerank(G)

# ---------------- TF-IDF ----------------
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(documents)

def tfidf_search(query):
    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, tfidf_matrix)[0]
    return {i: scores[i] for i in range(len(scores))}

# ---------------- HYBRID ----------------
def normalize(scores):
    max_val = max(scores.values())
    if max_val == 0:
        return {k: 0 for k in scores}
    return {k: v / max_val for k, v in scores.items()}


def hybrid_ranking(tfidf_scores, pagerank_scores, w_text=0.6, w_pr=0.4):
    tfidf_norm = normalize(tfidf_scores)
    pr_norm = normalize(pagerank_scores)

    hybrid = {}
    for k in tfidf_norm:
        hybrid[k] = w_text * tfidf_norm[k] + w_pr * pr_norm.get(k, 0)
    return hybrid
