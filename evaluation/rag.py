import ast

from llm_requests import extract_metadata, generate_reasoning
from pinecone_db import VectorDB
from trulens_eval.tru_custom_app import instrument


class RAG:
    def __init__(self, db: VectorDB) -> None:
        self.db = db

    @instrument
    def retrieve(self, query: str) -> str:
        """
        Retrieve relevant context from vector store.
        """
        metadata = extract_metadata(query)['generated_text']
        metadata = metadata.replace('{{', '{').replace('}}', '}')
        start = metadata.find('{')
        end = metadata.find('}')
        metadata = metadata[start:end+1].strip()
        metadata = ast.literal_eval(metadata)

        retrieved = self.db.search_movies(
            query,
            metadata,
            k=1, min_similarity_score=0)
        title, context = retrieved[0]['title'], retrieved[0]['description']

        self.db.search_movies(query, metadata, k=1)

        context = f'{title}\n' + context
        return context

    @instrument
    def generate_completion(self, query: str, context_str: list) -> str:
        """
        Generate reasoning from context.
        """
        title = context_str.split('\n')[0]
        context_str = '\n'.join(context_str.split('\n')[1:])
        reasoning = generate_reasoning(
            title=title,
            description=context_str,
            query=query
        )
        return reasoning['generated_text']

    @instrument
    def query(self, query: str) -> str:
        context_str = self.retrieve(query)
        completion = self.generate_completion(query, context_str)
        return completion
