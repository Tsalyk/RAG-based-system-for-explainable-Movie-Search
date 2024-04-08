from sentence_transformers import SentenceTransformer

from populate_db import init_db


class VectorDB:
    def __init__(self) -> None:
        self.conn = init_db()
        self.embedders = {
            'all-MiniLM-L6-v2': SentenceTransformer('all-MiniLM-L6-v2'),
            'gtr-t5-base': SentenceTransformer('gtr-t5-base'),
            'bert-base-nli-mean-tokens': SentenceTransformer('bert-base-nli-mean-tokens')
        }

    def get_embedding(self, text: str, embedding_model: str) -> list:
        embeddings = self.embedders[embedding_model].encode(
                                                text, convert_to_tensor=False
                                                )
        return embeddings.tolist()

    def search_movies(
            self,
            chunking_strategy: str,
            embedding_model: str,
            query: str,
            metadata: dict,
            k: int,
            min_similarity_score: float) -> list:
        cur = self.conn.cursor()
        query_vector = self.get_embedding(query, embedding_model)
        query_vector = list(map(str, query_vector))
        query_vector = f'ARRAY[{",".join(query_vector)}]'
        table_name = f"{chunking_strategy.lower().replace('-', '_')}_{embedding_model.lower().replace('-', '_')}"
        min_year = str(metadata.get('min_year', '0'))
        if min_year.isnumeric():
            min_year = int(min_year)
        else:
            min_year = 0
        max_year = str(metadata.get('max_year', '2025'))
        if max_year.isnumeric():
            max_year = int(max_year)
        else:
            max_year = 2015
        genre = metadata.get('genre', '')
        genre = genre if len(genre) > 0 else None
        filters = "WHERE 1 = 1"
        filters += f" AND year >= {min_year}"
        filters += f" AND year <= {max_year}"

        if genre:
            filters += f" AND '{genre}' = ANY(genres)"
        filters += f" AND 1 - (feature_vector::vector <=> {query_vector}::vector) >= {min_similarity_score}"

        sql_query = f"""
        SELECT id, title, year, genres, description, 1 - (feature_vector::vector <=> {query_vector}::vector) AS similarity
        FROM {table_name}
        {filters}
        ORDER BY similarity DESC
        LIMIT {k*5};
        """
        try:
            cur.execute(sql_query)
            results = cur.fetchall()
        except Exception as e:
            print(e)
            self.conn = init_db()
            results = []
        output = [{
            'title': row[1],
            'year': row[2],
            'genres': row[3],
            'description': row[4],
            'similarity': row[5]
        } for row in results]
        cur.close()

        return output
