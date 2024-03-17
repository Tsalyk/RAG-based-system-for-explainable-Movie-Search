import numpy as np
import pandas as pd


def extract_year(title: str) -> int:
    try:
        return int(title.split()[-1][1:-1])
    except Exception:
        return None


def clean_title(title: str) -> str:
    return ' '.join(title.split()[:-1])


def preprocess_movie_lens(movies_ml: pd.DataFrame) -> pd.DataFrame:
    movies_ml['year'] = movies_ml['title'].apply(extract_year)
    movies_ml = movies_ml.dropna(subset=['year', 'description'])
    movies_ml['year'] = movies_ml['year'].astype(int)
    movies_ml['genres'] = movies_ml['genres'].str.split('|')
    movies_ml['genres'] = movies_ml['genres'].apply(lambda x: [genre.strip() for genre in x])
    movies_ml['title'] = movies_ml['title'].apply(clean_title)
    movies_ml = movies_ml[['title', 'genres', 'year', 'description']]
    return movies_ml


def clean_netflix_genre(genre: str):
    return genre.replace('TV', '').replace('Series', '').replace('Movies', '').replace('Anime Features', 'Anime').strip()


def preproces_netflix(netflix: pd.DataFrame) -> pd.DataFrame:
    netflix = netflix.rename(columns={'release_year': 'year', 'listed_in': 'genres'})
    netflix['genres'] = netflix['genres'].apply(clean_netflix_genre)
    netflix = netflix[['title', 'genres', 'year', 'description']]
    netflix['genres'] = netflix['genres'].str.split(',')
    netflix['genres'] = netflix['genres'].apply(lambda x: [genre.strip() for genre in x])
    return netflix
