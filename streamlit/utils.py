import numpy as np
import pandas as pd

from api_requests import search_movies
from db_initialization import init_db


def streamlit_search_movies(
        chunking_strategy: str,
        embedding_model: str,
        query: str,
        metadata: dict,
        k=10, min_similarity_score=0) -> pd.DataFrame:

    result = search_movies(
        chunking_strategy=chunking_strategy,
        embedding_model=embedding_model,
        query=query,
        metadata=metadata,
        k=k,
        min_similarity_score=min_similarity_score)
    result = result['search_results']

    data = pd.DataFrame(columns=['title', 'year', 'genres', 'rag_description'])
    for movie in result:
        title, genres, year, description = movie['title'], movie['genres'], movie['year'], movie['description']
        genres = genres.replace('{', '').replace('}', '')
        if len(genres) == 0:
            genres = 'None'
        else:
            genres = ', '.join(genres.split(','))

        new_row = pd.DataFrame({
                'title': [title],
                'year': [year],
                'genres': [genres],
                'rag_description': [description]
            })

        if len(data) == 0:
            data = new_row
        else:
            data = pd.concat([data, new_row])

    data = data.drop_duplicates(subset=['title'])

    return data


def remove_emojis(text: str) -> str:
    for emoji in ['🇬🇧', '🇺🇦', '🇪🇸', '🇫🇷', '🇩🇪', '🇮🇹']:
        text = text.replace(emoji, '')
    return text


def count_empty_tables_proportion() -> float:
    CHUNKING_STRATEGIES = [
        'fixed-size-splitter',
        'recursive-splitter',
        'semantic-splitter'
    ]
    EMBEDDING_MODELS = [
        'all-MiniLM-L6-v2',
        'bert-base-nli-mean-tokens',
        'gtr-t5-base'
    ]
    conn = init_db()
    k_tables = len(CHUNKING_STRATEGIES) * len(EMBEDDING_MODELS)
    counter = 0

    for chunking_strategy in CHUNKING_STRATEGIES:
        for embedding_model in EMBEDDING_MODELS:
            table_name = f"{chunking_strategy.lower().replace('-', '_')}_{embedding_model.lower().replace('-', '_')}"
            n = count_rows(conn, table_name)

            if n < 10:
                counter += 1
    return round(counter/k_tables, 2)


def count_rows(conn, table_name: str) -> int:
    sql = f"SELECT COUNT(*) FROM {table_name}"
    cur = conn.cursor()
    cur.execute(sql)
    row_count = cur.fetchone()[0]
    cur.close()
    return row_count
