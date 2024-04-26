import ast
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv

import streamlit as st
from api_requests import extract_metadata, generate_reasoning, translate, is_db_api_alive
from utils import streamlit_search_movies, remove_emojis, count_empty_tables_proportion
import time

load_dotenv()


@st.cache_data
def load_data():
    DATASET = os.getenv('DATASET')
    df = pd.read_csv(DATASET)
    df['genres'] = df['genres'].apply(ast.literal_eval)
    return df


def display_population_progress():
    progress_text = "Data Indexing in progress! Please, wait..."
    bar = st.progress(0, text=progress_text)
    time_limit = 0

    if not st.session_state.get('db_ready', False):
        while time_limit < 20000:
            progress = int((1 - count_empty_tables_proportion())*100)
            bar.progress(progress, text=progress_text)
            time_limit += 5

            if progress > 99:
                api_status = is_db_api_alive().get('status', '')
                if api_status == 'alive':
                    break
                else:
                    progress_text = "API is launching! Please, wait..."
                    bar.progress(100, text=progress_text)
            time.sleep(5)

        bar.empty()
        st.session_state['db_ready'] = True


def display_movies(
        movies: pd.DataFrame,
        query: str,
        language: str,
        counter=0
        ):
    counter = st.session_state.get('counter', counter)

    prev_query = st.session_state.get('prev_query', '')
    if query != prev_query:
        st.session_state['counter'] = 0
        counter = 0

    movie = movies.iloc[counter]

    rag_description = movie['rag_description']

    title = translate(
        text=movie['title'].split(',')[0],
        src_lang="eng",
        tgt_lang=remove_emojis(language)
    )['translation']

    genres = translate(
        text=movie['genres'],
        src_lang="eng",
        tgt_lang=remove_emojis(language)
    )['translation']

    text = f"**Title:** {title}\n\n"
    text += f"**Genre:** {genres}\n\n"
    text += f"**Year:** {movie['year']}\n\n"

    st.write(text)

    with st.spinner('Generating reasoning...'):
        reasoning = generate_reasoning(
            movie['title'], rag_description, query
            )
        if 'generated_response' not in reasoning:
            reasoning = {'generated_response': None}

        text = "**🤖AI Reasoning**\n\n"
        reasoning = translate(
            text=reasoning['generated_response'],
            src_lang="en",
            tgt_lang=remove_emojis(language)
        )['translation']
        text += reasoning

        found_text = "Movie is found!"
        found_text = translate(
            text=found_text,
            src_lang="eng",
            tgt_lang=remove_emojis(language)
        )['translation']
        st.success(found_text)

        st.write(text)
        st.write("---")

    if st.button(f'Show More ({counter+1}/{len(movies)})', key=f"show_more_{counter}"):
        st.session_state['counter'] = counter+1
        st.session_state['prev_query'] = query
        st.rerun()
        display_movies(movies, query, language)


def main():
    data = load_data()

    chunking_strategy = st.sidebar.selectbox(
                                            'Select Chunking strategy',
                                            [
                                                'fixed-size-splitter',
                                                'recursive-splitter',
                                                'semantic-splitter'
                                                ]
                                                )
    embedding_model = st.sidebar.selectbox(
                                            'Select Embedding model',
                                            [
                                                'all-MiniLM-L6-v2',
                                                'bert-base-nli-mean-tokens',
                                                'gtr-t5-base'
                                                ]
                                                )
    language = st.sidebar.selectbox(
                                        'Select Generation Output Language',
                                        [
                                            '🇬🇧en',
                                            '🇺🇦uk',
                                            '🇪🇸es',
                                            '🇫🇷fr',
                                            '🇩🇪de',
                                            '🇮🇹it'
                                            ]
                                            )

    st.title('Movie Search App')
    st.session_state.setdefault('counter', 0)
    st.session_state.setdefault('prev_query', '')
    st.session_state.setdefault('db_ready', False)

    disabled = True

    db_ready = st.session_state.get('db_ready', False)

    if not db_ready:
        disabled = True
        search_query = st.text_input(
            'Enter movie title, genre, or description:',
            key='disabled_search',
            disabled=disabled
            )
        display_population_progress()
        disabled = False
        st.rerun()
    else:
        disabled = False

    if not disabled:
        search_query = st.text_input(
            'Enter movie title, genre, or description:',
            key='enabled_search',
            disabled=disabled
            )
        search_query = translate(
            text=search_query,
            src_lang=remove_emojis(language),
            tgt_lang="en"
        )['translation']

        movies, metadata = [], {'generated_response': ''}
        if search_query:
            metadata = extract_metadata(search_query)
            try:
                metadata = ast.literal_eval(metadata['generated_response'])
            except Exception:
                metadata = {}
            movies = streamlit_search_movies(
                chunking_strategy, embedding_model, search_query, metadata
                )

        if len(movies) > 0:
            display_movies(movies, search_query, language)
        else:
            st.write('Try to search for some movies with description')


if __name__ == "__main__":
    main()
