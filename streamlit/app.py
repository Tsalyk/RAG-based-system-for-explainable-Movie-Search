import ast
import os
import time

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from llm_requests import *
from pinecone import Pinecone
from pinecone_db import *
from sentence_transformers import SentenceTransformer

import streamlit as st

load_dotenv()


def search_movies(db, query: str, metadata: pd.DataFrame, data: pd.DataFrame, k=10, min_similarity_score=0) -> pd.DataFrame:
    result = db.search_movies(query, metadata, k=k, min_similarity_score=min_similarity_score)
    titles = np.unique([movie['title'] for movie in result])
    titles, descriptions = [], []
    for movie in result:
        title, description = movie['title'], movie['description']
        if title not in titles:
            titles.append(title)
            descriptions.append(description)

    data = data[data['title'].isin(titles)]
    data['rag_description'] = None

    for title, description in zip(titles, descriptions):
        data.loc[data['title'] == title, 'rag_description'] = description

    return data

def stream_reasoning(reasoning: str):
    for word in reasoning.split():
        yield word + " "
        time.sleep(0.02)

def display_movies(movies, query, metadata, batch_size=2):
    batch_n = st.session_state.get('batch', 0)
    num_batches = len(movies) // batch_size

    for i in range(batch_n, num_batches + 1):
        batch = movies[i * batch_size:(i + 1) * batch_size]

        for _, movie in batch.iterrows():
            rag_description = movie['rag_description']
            reasoning = generate_reasoning(movie['title'], rag_description, query)

            if 'generated_text' not in reasoning:
                print(movie['description'])
                print(reasoning)

            st.write(f"**Title:** {movie['title']}")
            st.write(f"**Genre:** {', '.join(movie['genres'])}")
            st.write(f"**Year:** {movie['year']}")
            st.write(f"**Description:** {movie['description']}...")
            st.write(f"**🤖AI Reasoning**")
            st.write_stream(stream_reasoning(reasoning['generated_text']))
            st.write(f"**Extracted metadata:** {metadata}")
            st.write("---")

        if st.button(f'Show More ({i+1}/{num_batches})', key=f"show_more_{i}"):
            st.session_state['batch'] += 1
            continue
        else:
            break

def init_db():
    PINECONE_KEY = os.getenv('PINECONE_API_KEY')
    db = VectorDB(PINECONE_KEY)
    db.set_index()
    return db

def main():
    DATASET = os.getenv('DATASET')
    db = init_db()
    data = pd.read_csv(DATASET)
    data['genres'] = data['genres'].apply(ast.literal_eval)

    st.title('Movie Search App')
    st.session_state.setdefault('batch', 0)

    search_query = st.text_input('Enter movie title, genre, or description:')

    movies, metadata = [], {}
    if search_query:
        metadata = extract_metadata(search_query)
        movies = search_movies(db, search_query, {}, data)
    # movies = db.search_movies(search_query, {})

    if len(movies) > 0:
        display_movies(movies, search_query, metadata)
    else:
        st.write('Try to search for some movies with description')


if __name__ == "__main__":
    main()
