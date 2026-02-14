import os
import faiss
import numpy as np
import pickle
from typing import List, Any
from sentence_transformers import SentenceTransformer
from embedding import EmbeddingPipeline
from data_loader import loaders

class FaissVectorStore:
    def __init__(self, persist_dir:str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap : int = 200):
        self.persist_dir = persist_dir
        self.embedding_model = SentenceTransformer(embedding_model)
        self.index = None
        if not os.path.exists(persist_dir):
            os.makedirs(persist_dir)
        self.metadata = []
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print(f"[INFO] Initialized FaissVectorStore with directory: {persist_dir} and model : {embedding_model}")
        
    def build_from_documents(self, documents: List[Any]):
        print(f"[INFO] Building vector store from {len(documents)} raw documents......")
        embedding_pipeline = EmbeddingPipeline(self.embedding_model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = embedding_pipeline.chunk_documents(documents)
        embeddings = embedding_pipeline.embed_chunks(chunks)
        metadatas = [{"text": chunk} for chunk in chunks]
        self.add_embeddings(np.array(embeddings).astype("float32"), metadatas)
        self.save()
        print(f"[INFO] Vector store built with {len(chunks)} chunks and saved to {self.persist_dir}")
        
    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any]):
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        print(f"[INFO] Added {embeddings.shape[0]} embeddings to the Faiss index.")
    
    def save(self):
        faiss_path = os.path.join(self.persist_dir, "faiss_index.bin")
        faiss.write_index(self.index, faiss_path)
        metadata_path = os.path.join(self.persist_dir, "metadata.pkl")
        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[INFO] Saved Faiss index to {faiss_path} and metadata to {metadata_path}")
        
    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss_index.bin")
        if os.path.exists(faiss_path):
            self.index = faiss.read_index(faiss_path)
        metadata_path = os.path.join(self.persist_dir, "metadata.pkl")
        if os.path.exists(metadata_path):
            with open(metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
        print(f"[INFO] Loaded Faiss index from {faiss_path} and metadata from {metadata_path}")
        
    def search(self, query: str, top_k: int = 5) -> List[Any]:
        query_embedding = self.embedding_model.encode([query]).astype("float32")
        D, I = self.index.search(query_embedding, top_k)
        results = []
        # Traversing index and distances in parallel from search results
        for idx, dist in zip(I[0], D[0]):
            if idx < len(self.metadata):
                meta = self.metadata[idx] if idx < len(self.metadata) else None
                results.append({"index": idx, 
                                "metadata": meta, 
                                "distance": dist}
                               )
        return results
    
    def query(self, query: str, top_k: int = 5) -> List[Any]:
        print(f"[INFO] Querying vector store for: {query}")
        results = self.search(query, top_k)
        return results
    
if __name__ == "__main__":
    # documents = loaders.load_all_documents("data")
    vector_store = FaissVectorStore()
    # vector_store.build_from_documents(documents)
    vector_store.load()
    results = vector_store.query("Give me 2nd semester details", top_k=3)
    for res in results:
        print(res)
        