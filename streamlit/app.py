import ast
import os
import time

import numpy as np
import pandas as pd
from api_requests import extract_metadata, generate_reasoning
from dotenv import load_dotenv
from utils import streamlit_search_movies

import streamlit as st

load_dotenv()


@st.cache_data
def load_data():
    DATASET = os.getenv('DATASET')
    df = pd.read_csv(DATASET)
    df['genres'] = df['genres'].apply(ast.literal_eval)
    return df


def display_movies(movies, query, metadata, batch_size=2):
    batch_n = st.session_state.get('batch', 0)
    num_batches = len(movies) // batch_size

    for i in range(batch_n, num_batches + 1):
        batch = movies[i * batch_size:(i + 1) * batch_size]

        for _, movie in batch.iterrows():
            rag_description = movie['rag_description']
            reasoning = generate_reasoning(
                movie['title'], rag_description, query)

            if 'generated_response' not in reasoning:
                reasoning = {'generated_response': None}

            st.write(f"**Title:** {movie['title']}")
            st.write(f"**Genre:** {', '.join(movie['genres'])}")
            st.write(f"**Year:** {movie['year']}")
            st.write(f"**Description:** {movie['description']}...")
            st.write("**🤖AI Reasoning**")
            st.write_stream(reasoning['generated_response'])
            st.write("---")

        if st.button(f'Show More ({i+1}/{num_batches})', key=f"show_more_{i}"):
            st.session_state['batch'] += 1
            continue
        else:
            break


def main():
    data = load_data()

    st.title('Movie Search App')
    st.session_state.setdefault('batch', 0)

    search_query = st.text_input('Enter movie title, genre, or description:')

    movies, metadata = [], {'generated_response': ''}
    if search_query:
        metadata = extract_metadata(search_query)
        metadata = ast.literal_eval(metadata['generated_response'])
        movies = streamlit_search_movies(search_query, metadata, data)
    # movies = db.search_movies(search_query, {})

    if len(movies) > 0:
        display_movies(movies, search_query, metadata)
    else:
        st.write('Try to search for some movies with description')


if __name__ == "__main__":
    main()
