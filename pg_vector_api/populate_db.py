import ast
import os

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()


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


def init_db():
    HOST = os.getenv('HOST')
    USER = os.getenv('USER')
    PASSWORD = os.getenv('PASSWORD')

    # HOST = 'localhost'
    # HOST = 'pgvector'
    # USER = 'admin'
    # PASSWORD = 'admin'

    conn = psycopg2.connect(
                        host=HOST,
                        user=USER,
                        password=PASSWORD
                        )
    return conn


def load_data(chunking_strategy: str, embedding_model: str) -> pd.DataFrame:
    path = f"embeddings/{chunking_strategy}/{embedding_model}/embeddings.csv"
    embeddings_df = pd.read_csv(path)
    embeddings_df['genres'] = embeddings_df['genres'].apply(
                                                        ast.literal_eval
                                                        )
    embeddings_df['embedding'] = embeddings_df['embedding'].apply(
                                                        ast.literal_eval
                                                        )
    return embeddings_df


def insert_row(
        conn,
        table_name: str,
        title: str,
        year: int,
        genres: list,
        description: str,
        vector: list):
    genres = "{" + ",".join(genres) + "}"
    query = f"""
INSERT INTO {table_name}
(title, year, genres, description, feature_vector) VALUES (%s, %s, %s, %s, %s);
            """
    cur = conn.cursor()
    cur.execute(query, (title, year, genres, description, vector))
    conn.commit()
    cur.close()


def populate_df(conn, table_name: str, df: pd.DataFrame):
    for i, (_, row) in zip(tqdm(range(len(df))), df.iterrows()):
        title = row['title']
        year = row['year']
        genres = row['genres']
        description = row['description']
        vector = row['embedding']
        insert_row(conn, table_name, title, year, genres, description, vector)


def delete_content(conn, table_name: str):
    sql = f"DELETE FROM {table_name}"
    cur = conn.cursor()
    cur.execute(sql)
    conn.commit()
    cur.close()


def count_rows(conn, table_name: str) -> int:
    sql = f"SELECT COUNT(*) FROM {table_name}"
    cur = conn.cursor()
    cur.execute(sql)
    row_count = cur.fetchone()[0]
    cur.close()
    return row_count


def populate_all():
    conn = init_db()

    for chunking_strategy in CHUNKING_STRATEGIES:
        for embedding_model in EMBEDDING_MODELS:
            embeddings_df = load_data(chunking_strategy, embedding_model)
            table_name = f"{chunking_strategy.lower().replace('-', '_')}_{embedding_model.lower().replace('-', '_')}"
            n = count_rows(conn, table_name)
            if n < 10:
                delete_content(conn, table_name)
                populate_df(conn, table_name, embeddings_df)


if __name__ == '__main__':
    populate_all()
