import ast

import pandas as pd
from llm_requests import extract_metadata, generate_reasoning, search_movies
from trulens_eval.tru_custom_app import instrument


class RAG:
    def __init__(self, chunking_strategy: str, embedding_model: str) -> None:
        self.chunking_strategy = chunking_strategy
        self.embedding_model = embedding_model

    @instrument
    def retrieve(self, query: str) -> str:
        """
        Retrieve relevant context from vector store.
        """
        metadata = extract_metadata(query)
        metadata = ast.literal_eval(metadata['generated_response'])

        retrieved = search_movies(
            chunking_strategy=self.chunking_strategy,
            embedding_model=self.embedding_model,
            query=query,
            metadata=metadata,
            k=1,
            min_similarity_score=0
            )['search_results']

        if len(retrieved) == 0:
            retrieved = search_movies(
                chunking_strategy=self.chunking_strategy,
                embedding_model=self.embedding_model,
                query=query,
                metadata={},
                k=1,
                min_similarity_score=0
                )['search_results']

        title, context = retrieved[0]['title'], retrieved[0]['description']
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
        return reasoning['generated_response']

    @instrument
    def query(self, query: str) -> str:
        context_str = self.retrieve(query)
        completion = self.generate_completion(query, context_str)
        return completion
