import streamlit as st
import pandas as pd
import numpy as np

import ast
import os

from dotenv import load_dotenv

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from pinecone_db import *

import requests

load_dotenv()


def search_movies(db, query: str, metadata: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    result = db.search_movies(query, metadata, k=10, min_similarity_score=0)
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

def extract_metadata(query: str):
    url = 'https://randomly-excited-gnat.ngrok-free.app/extract_metadata'
    body = {
        "query": query,
        "parameters": {}
    }
    try:
        response = requests.post(url, json=body)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Request failed with status code {response.status_code}"}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

def generate_reasoning(title: str, description: str, query: str):
    url = 'https://randomly-excited-gnat.ngrok-free.app/generate_reasoning'
    body = {
        "title": title,
        "query": query,
        "description": description,
        "parameters": {}
    }
    try:
        response = requests.post(url, json=body)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Request failed with status code {response.status_code}"}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

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
            st.write(f"**Reasoning:** {reasoning['generated_text']}")
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
    db = init_db()
    data = pd.read_csv('streamlit/data.csv')
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
