import uuid
import warnings

from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

warnings.filterwarnings('ignore')
import os

load_dotenv()


class VectorDB:
    def __init__(self, api_key: str, dimension: int = 384):
        self.pc = Pinecone(api_key)
        self.vector_dim = dimension
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.movie_genres = self.get_all_genres()

    def get_all_genres(self) -> list:
        with open(os.getenv('GENRES'), 'r') as f:
            genres = f.readlines()
            genres = list(map(lambda g: g.strip(), genres))
        return genres

    def create_index(self, index_name: str, metric: str = 'cosine'):
        self.pc.create_index(
            index_name,
            dimension=self.vector_dim,
            metric=metric,
            spec=ServerlessSpec(
                    cloud="aws",
                    region="us-west-2"
                )
            )

    def get_movies_index(self):
        return self.pc.Index("movies")

    def set_index(self):
        self.index = self.get_movies_index()

    def upsert_movie(self, title: str, genres: list, year: int, description: str):
        MAX_RETRIES = 10
        year = int(year)
        descriptions = self.chunk_description(description)

        # genres preproccessing
        genres = list(filter(lambda g: g not in ['',  '(no genres listed)'] and '&' not in g, genres))
        genres = list(map(lambda g: g.replace('  ', ' ').lower(), genres))
        genres = self.encode_genres(genres)

        metadata = {'title': title, 'year': year}
        metadata = metadata | genres
        movie_id = str(uuid.uuid4())

        retries = 0
        for description in descriptions:
            record_id = str(uuid.uuid4())
            embeddings = self.get_embedding(description)
            upserted = False
            while not upserted and retries < MAX_RETRIES:
                try:
                    self.index.upsert([(record_id, embeddings, metadata |
                                        {'description': description,
                                         'movie_id': movie_id})])
                    upserted = True
                    retries = 0
                except Exception:
                    retries += 1

    def search_movies(
            self, query: str,
            metadata: dict,
            k=10, min_similarity_score=0.25) -> list:
        query_embedding = self.get_embedding(query)

        # Applying filters
        filters = {}
        genre = metadata.get('genre', '')
        if genre:
            filters = filters | self.encode_genres([genre], leave_zeros=False)

        min_year = metadata.get('min_year', '0')
        if len(min_year) > 0:
            min_year = int(min_year)
        else:
            min_year = 0
        max_year = metadata.get('max_year', '2030')
        if len(max_year) > 0:
            max_year = int(max_year)
        else:
            max_year = 2030

        # years = list(range(min_year, max_year+1))

        # filters['year'] = {
        #     '$gte': min_year,
        #     # '$lte': max_year
        #     }
        # filters['$and'] = [{'year': {'$gte': min_year}}, {'year': {'$lte': max_year}}]
        # filters['year'] = {'$in': years}

        search_results = self.index.query(
                    vector=query_embedding,
                    filter=filters,
                    top_k=k,
                    include_values=False
                    )

        files = [{
            'movie_id': self.get_movie_id(search['id']),
            'title': self.get_movie_title(search['id']),
            'genres': self.get_movie_genres(search['id']),
            'description': self.get_movie_description(search['id']),
            'score': search['score']
            } for search in search_results['matches'] if search['score'] > min_similarity_score]

        return files[:k]

    def encode_genres(self, genres: list, leave_zeros=True) -> dict:
        encoded_genres = {g: 0 for g in self.movie_genres if leave_zeros}
        for g in genres:
            encoded_genres[g] = 1
        return encoded_genres

    def get_embedding(self, text: str) -> list:
        embeddings = self.embedder.encode(text, convert_to_tensor=False)
        return embeddings.tolist()

    @staticmethod
    def chunk_description(description: str) -> list:
        text_splitter = CharacterTextSplitter(
                            separator="\n",
                            chunk_size=1024,
                            chunk_overlap=20
                        )
        docs = text_splitter.create_documents([description])
        docs = [str(doc) for doc in docs]
        return docs

    def get_movie_title(self, record_id: str) -> list:
        return self.get_metadata(record_id).get('title', '')

    def get_movie_genres(self, record_id: str) -> list:
        return self.get_metadata(record_id).get('genres', '')

    def get_movie_description(self, record_id: str) -> list:
        return self.get_metadata(record_id).get('description', '')

    def get_movie_id(self, record_id: str) -> list:
        return self.get_metadata(record_id).get('movie_id', '')

    def get_record_id(self, title: str) -> list:
        metadata = {'title': title}
        sr = self.index.query(
                    vector=list(range(self.vector_dim)),
                    filter=metadata,
                    top_k=1,
                    include_values=False
                    )
        return sr['matches'][0]['id']

    def get_metadata(self, file_id: str) -> dict:
        return self.index.fetch([file_id]).vectors[file_id]['metadata']

    def delete_file(self, filename: str, space_id: str, tenant_id: str):
        file_id = self.get_file_id(filename, space_id, tenant_id)
        self.index.delete(ids=[file_id], namespace='')

    def remove_index(self, index_name: str):
        self.pc.delete_index(index_name)


def init_db():
    PINECONE_KEY = os.getenv('PINECONE_API_KEY')
    db = VectorDB(PINECONE_KEY)
    db.set_index()
    return db


if __name__ == '__main__':
    db = init_db()
