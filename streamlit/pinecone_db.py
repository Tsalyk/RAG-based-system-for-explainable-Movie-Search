from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

tqdm.pandas()

import ast
import uuid
import warnings

from langchain.text_splitter import CharacterTextSplitter

warnings.filterwarnings('ignore')
import os


class VectorDB:
    def __init__(self, api_key: str, dimension: int=384):
        # pinecone.init(api_key=api_key, environment=environment)
        self.pc = Pinecone(api_key)
        self.vector_dim = dimension
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def create_index(self, index_name: str, metric: str='cosine'):
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
        year = int(year)
        descriptions = self.chunk_description(description)
        genres = list(map(lambda g: g.lower(), genres))
        metadata = {'title': title, 'year': year, 'genres': genres}
        movie_id = str(uuid.uuid4())

        for description in descriptions:
            record_id = str(uuid.uuid4())
            embeddings = self.get_embedding(description)
            upserted = False
            while not upserted:
                try:
                    self.index.upsert([(record_id, embeddings, metadata | {'description': description, 'movie_id': movie_id})])
                    upserted = True
                except:
                    pass

    def search_movies(self, query: str, metadata: dict, k=10, min_similarity_score=0.25) -> list:
        query_embedding = self.get_embedding(query)

        # Applying filters
        filters = {}
        genre = metadata.get('genre', '')
        if genre:
            filters['genre'] = genre.lower()
        year = metadata.get('year', '')
        if len(str(year)) > 0:
            try:
                filters['year'] = int(year)
            except:
                pass

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



if __name__ == '__main__':
    PINECONE_KEY = os.getenv('PINECONE_API_KEY')
    db = VectorDB(PINECONE_KEY)
    db.set_index()
