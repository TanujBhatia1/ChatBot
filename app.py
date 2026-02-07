from src.data_loader import loaders
from src.embedding import EmbeddingPipeline
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

if __name__ == "__main__":
    documents = loaders.load_all_documents("data")
    store = FaissVectorStore("faiss_store")
    # store.build_from_documents(documents)
    store.load()
    # print(store.query("What is the total marks scored by Tanuj in 8th semester?", top_k=5))
    search = RAGSearch()
    print("Search initialized with loaded vector store.")
    # query = "What is the score of Tanuj in Gate 2025?"
    query = input("Enter your query: ")
    answer = search.search(query)
    print(f"Answer: {answer}")