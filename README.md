# Hypertextual Web Search Engine (Experimental Prototype)

This project is a small-scale experimental implementation inspired by
"The Anatomy of a Large-Scale Hypertextual Web Search Engine" by Brin and Page.

## Features
- TF-IDF based keyword ranking
- PageRank based authority ranking
- Hybrid ranking combining content relevance and link structure
- Synthetic web graph generation
- Interactive frontend using Streamlit

## Tech Stack
- Python
- NetworkX
- Scikit-learn
- Streamlit

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
