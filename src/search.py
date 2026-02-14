import os
from typing import List, Dict
from dotenv import load_dotenv
from vectorstore import FaissVectorStore
from conversation_memory import ConversationMemory
from langchain_groq import ChatGroq

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "llama-3.1-8b-instant"):
        self.vector_store = FaissVectorStore(persist_dir=persist_dir, embedding_model=embedding_model)
        self.memory = ConversationMemory(max_history=10)
        
        faiss_path = os.path.join(persist_dir, "faiss_index.bin")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from data_loader import loaders
            documents = loaders.load_all_documents("../data")
            self.vector_store.build_from_documents(documents)
        else:
            self.vector_store.load()
            
        groq_api_key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(api_key=groq_api_key, model=llm_model)
        print(f"[INFO] Initialized RAGSearch with LLM model: {llm_model} and embedding model: {embedding_model}")
        
    def search(self, query: str, top_k: int = 5) -> str:
        """Search with conversational context"""
        results = self.vector_store.search(query, top_k=top_k)
        print(f"[DEBUG] Retrieved {len(results)} results from vector store for query: {query}")
        texts = [res["metadata"].get("text", "") for res in results]
        context = "\n\n".join(texts)
        
        if not context:
            response_text = "No relevant information found in the documents."
        else:
            # Include conversation history in prompt
            conversation_context = self.memory.get_context()
            
            prompt = f"""You are a helpful and friendly assistant. Answer the question based on the provided context.
                        Be conversational and maintain continuity with previous exchanges. If you're unsure about something, admit it.
                        Keep your response concise and relevant.

                        {conversation_context}

                        Context from documents:
                        {context}

                        Current Question: {query}
                        Answer:
                        """
            
            response = self.llm.invoke(prompt)
            response_text = response.content
        
        # Store in conversation memory
        self.memory.add_exchange(query, response_text)
        return response_text
    
    def get_conversation_history(self) -> List[Dict]:
        """Get full conversation history"""
        return self.memory.get_history()
    
    def clear_history(self):
        """Clear conversation history"""
        self.memory.clear()
    
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "can you tell me the total marks scored by Tanuj in 8th semester ?"
    answer = rag_search.search(query)
    print(f"Answer: {answer}")