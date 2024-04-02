from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from sentence_transformers import SentenceTransformer
from scripts.hg_embeddings_connector import CustomEmbedder


def fixed_size_chunking(description: str, embedder: SentenceTransformer) -> list:
    text_splitter = CharacterTextSplitter(
                        separator="\n",
                        chunk_size=1024,
                        chunk_overlap=20
                    )
    docs = text_splitter.split_text(description)
    return docs


def semantic_chunking(description: str, embedder: SentenceTransformer) -> list:
    emb = CustomEmbedder(embedder)
    text_splitter = SemanticChunker(emb)
    docs = text_splitter.split_text(description)
    return docs


def recursive_chunking(description: str, embedder: SentenceTransformer) -> list:
    text_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=1024,
                                chunk_overlap=20,
                                length_function=len,
                                is_separator_regex=False,
                            )
    docs = text_splitter.split_text(description)
    return docs
