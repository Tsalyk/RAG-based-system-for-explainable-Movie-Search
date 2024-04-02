class CustomEmbedder:
    def __init__(self, embedder) -> None:
        self.embedder = embedder

    def get_embedding(self, text: str) -> list:
        embeddings = self.embedder.encode(text, convert_to_tensor=False)
        return embeddings.tolist()

    def embed_documents(self, documents: list) -> list:
        return [self.get_embedding(doc) for doc in documents]

    def embed_query(self, query: str) -> list:
        return self.get_embedding(query)
