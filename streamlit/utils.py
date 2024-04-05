import numpy as np
import pandas as pd

from api_requests import search_movies


def streamlit_search_movies(
        chunking_strategy: str,
        embedding_model: str,
        query: str,
        metadata: dict,
        data: pd.DataFrame,
        k=10, min_similarity_score=0) -> pd.DataFrame:

    result = search_movies(
        chunking_strategy=chunking_strategy,
        embedding_model=embedding_model,
        query=query,
        metadata=metadata,
        k=k,
        min_similarity_score=min_similarity_score)
    result = result['search_results']

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
