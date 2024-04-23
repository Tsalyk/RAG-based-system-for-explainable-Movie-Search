import numpy as np
import pandas as pd

from api_requests import search_movies


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
