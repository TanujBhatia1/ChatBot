from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from src.data_loader import loaders

class EmbeddingPipeline:
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        # self.model = SentenceTransformer(model_name)
        self.model = model_name if isinstance(model_name, SentenceTransformer) else SentenceTransformer(model_name)
        self.chunk_overlap = chunk_overlap
        print(f"[DEBUG] Loading embedding model: {model_name}")
        
    def chunk_documents(self, documents: List[Any]) -> List[str]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        all_chunks = []
        for doc in documents:
            chunks = text_splitter.split_text(doc.page_content)
            all_chunks.extend(chunks)
            print(f"[DEBUG] Document split into {len(chunks)} chunks.")
        return all_chunks

    def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        all_embeddings = []
        for chunk in chunks:
            embedding = self.model.encode(chunk)
            all_embeddings.append(embedding)
        return np.array(all_embeddings)