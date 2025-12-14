import streamlit as st
from model import tfidf_search, hybrid_ranking, pagerank_scores, documents

st.set_page_config(page_title="Hypertextual Search Engine", layout="centered")

st.title("🔍 Hypertextual Web Search Engine")
st.write("Semantic + Link-based Ranking Prototype")

query = st.text_input("Enter search query:")

if st.button("Search"):
    tfidf_scores = tfidf_search(query)
    hybrid_scores = hybrid_ranking(tfidf_scores, pagerank_scores)

    top_results = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:5]

    st.subheader("Top Results (Hybrid Ranking)")
    for doc_id, score in top_results:
        st.write(f"**Doc {doc_id}** | Score: {score:.4f}")
        st.caption(documents[doc_id])

