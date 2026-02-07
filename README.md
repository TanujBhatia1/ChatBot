# ChatBotTanuj - RAG-Powered Document Search Assistant

A sophisticated **Retrieval-Augmented Generation (RAG)** chatbot that enables intelligent document search and question-answering by combining vector embeddings with advanced language models. This application processes multiple document formats and provides context-aware answers using semantic search.

## Features

✨ **Multi-Format Document Support**
- PDF documents
- Text files (.txt)
- CSV spreadsheets
- Excel files (.xlsx)
- Word documents (.docx)
- JSON files

🔍 **Intelligent Document Processing**
- Automatic document chunking with configurable overlap
- Semantic embeddings using Sentence Transformers
- FAISS vector store for efficient similarity search

🤖 **Advanced Question-Answering**
- RAG pipeline combining retrieval and generation
- Groq API integration for fast LLM inference
- Context-aware responses based on document content

⚡ **High Performance**
- GPU-optimized embedding generation
- Fast vector similarity search with FAISS
- Efficient document indexing and storage

## Project Structure

```
ChatBotTanuj/
├── app.py                      # Main application entry point
├── main.py                     # Secondary entry point
├── requirements.txt            # Python dependencies
├── pyproject.toml             # Project metadata and configuration
├── README.md                  # This file
│
├── src/                       # Core source code
│   ├── __init__.py
│   ├── data_loader.py         # Document loading utilities
│   ├── embedding.py           # Embedding pipeline
│   ├── vectorstore.py         # FAISS vector store implementation
│   └── search.py              # RAG search implementation
│
├── data/                      # Document storage
│   ├── pdf/                   # PDF documents
│   ├── text_files/            # Text documents
│   └── vector_store/          # ChromaDB storage
│
├── faiss_store/               # FAISS index and metadata
│   ├── faiss_index.bin        # Serialized FAISS index
│   └── metadata.pkl           # Document metadata
│
└── notebook/                  # Jupyter notebooks
    ├── document.ipynb         # Document processing notebook
    └── pdf_to_VectorDb.ipynb  # Vector DB creation notebook
```

## Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   cd ChatBotTanuj
   ```

2. **Create and activate virtual environment** (optional but recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the project root:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```
   
   Get your Groq API key from [Groq Console](https://console.groq.com)

## Quick Start

### Basic Usage

```bash
python app.py
```

When prompted, enter your query:
```
Enter your query: What is the total marks scored by Tanuj in 8th semester?
```

The system will search through your documents and provide an answer based on the relevant context found.

### Programmatic Usage

```python
from src.search import RAGSearch

# Initialize RAG search
search = RAGSearch()

# Query the documents
query = "What are the details for 8th semester?"
answer = search.search(query, top_k=5)
print(answer)
```

## Core Components

### 1. Data Loader (`src/data_loader.py`)
Handles loading documents from various file formats with metadata enrichment.

**Features:**
- Automatic file type detection
- Recursive directory traversal
- Error handling and logging
- Source tracking in metadata

**Methods:**
- `load_all_documents(data_dir)` - Load all PDFs from directory
- `load_text_files(data_dir)` - Load text files
- `load_csv_files(data_dir)` - Load CSV files
- `load_docx_files(data_dir)` - Load Word documents

### 2. Embedding Pipeline (`src/embedding.py`)
Converts documents into semantic embeddings using Sentence Transformers.

**Features:**
- Configurable text chunking (default: 1000 characters)
- Overlap handling (default: 200 characters)
- Batch embedding generation
- NumPy array output

**Key Methods:**
- `chunk_documents(documents)` - Split documents into chunks
- `embed_chunks(chunks)` - Generate embeddings for chunks

**Default Model:** `all-MiniLM-L6-v2` (384-dimensional embeddings)

### 3. Vector Store (`src/vectorstore.py`)
Manages FAISS vector database for efficient similarity search.

**Features:**
- Persistent index storage
- Metadata preservation
- Configurable search parameters
- L2 distance metric

**Key Methods:**
- `build_from_documents(documents)` - Create index from raw documents
- `add_embeddings(embeddings, metadatas)` - Add vectors to index
- `search(query, top_k)` - Retrieve top-k similar documents
- `save()` / `load()` - Persist/restore index

### 4. RAG Search (`src/search.py`)
Implements Retrieval-Augmented Generation pipeline combining vector search with LLM inference.

**Features:**
- Automatic vector store initialization
- Context-based prompt generation
- Groq LLM integration for fast inference
- Fallback handling for empty results

**Key Methods:**
- `search(query, top_k)` - Execute RAG pipeline and return answer

## Configuration

### Customizing Embedding Model

```python
from src.search import RAGSearch

# Use a different embedding model
search = RAGSearch(embedding_model="all-mpnet-base-v2")
```

### Customizing LLM Model

```python
from src.search import RAGSearch

# Use a different Groq model
search = RAGSearch(llm_model="mixtral-8x7b-32768")
```

### Adjusting Chunk Size

```python
from src.vectorstore import FaissVectorStore

# Larger chunks = more context, fewer chunks
store = FaissVectorStore(chunk_size=2000, chunk_overlap=400)
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `langchain` | >=1.0.8 | LLM framework |
| `langchain-community` | >=0.4.1 | Document loaders |
| `langchain-groq` | >=1.0.1 | Groq integration |
| `sentence-transformers` | >=5.1.2 | Embedding models |
| `faiss-cpu` | >=1.13.0 | Vector similarity search |
| `pymupdf` | >=1.26.6 | PDF processing |
| `pypdf` | >=6.3.0 | PDF support |
| `chromadb` | >=1.3.5 | Vector database |
| `python-dotenv` | >=1.2.1 | Environment variables |

## API Keys

### Groq API
Required for LLM inference. Get your free API key:
1. Visit [Groq Console](https://console.groq.com)
2. Sign up or log in
3. Create API key
4. Add to `.env` file: `GROQ_API_KEY=your_key`

## Advanced Usage

### Building Vector Store from Scratch

```python
from src.data_loader import loaders
from src.vectorstore import FaissVectorStore

# Load documents
documents = loaders.load_all_documents("data")

# Create and build vector store
store = FaissVectorStore("faiss_store")
store.build_from_documents(documents)
```

### Custom Search with Direct Vector Store

```python
from src.vectorstore import FaissVectorStore

store = FaissVectorStore()
store.load()

# Search with top_k results
results = store.search("Your query here", top_k=10)

for result in results:
    print(f"Text: {result['metadata']['text']}")
    print(f"Distance: {result['distance']}")
```

## Performance Tips

1. **Use GPU acceleration** - Install `faiss-gpu` for faster similarity search
2. **Adjust chunk size** - Larger chunks process faster but may lose granularity
3. **Batch processing** - Process multiple documents together
4. **Cache embeddings** - Persist and reuse vector store to avoid recomputation

## Troubleshooting

| Issue | Solution |
|-------|----------|
| GROQ_API_KEY not found | Ensure `.env` file exists in project root with valid API key |
| "No relevant information found" | Ensure documents are loaded in `data/` directory and vector store is built |
| Slow performance | Consider using GPU with `faiss-gpu` or reduce `top_k` value |
| PDF loading errors | Verify PDF files are not corrupted and in `data/pdf/` directory |

## Example Queries

```
"What are the marks in semester 8?"
"Provide semester details"
"How many credits are in the program?"
"Show the academic transcript"
"What is the GPA breakdown?"
```

## Future Enhancements

- [ ] Web UI with Streamlit/Gradio
- [ ] Multi-language support
- [ ] Real-time document indexing
- [ ] Query result ranking and filtering
- [ ] Document source highlighting
- [ ] Batch query processing
- [ ] GraphQL API integration

## Contributing

Contributions are welcome! Please follow these guidelines:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For issues, questions, or suggestions, please create an issue in the repository or contact the maintainer.

---

**Built with ❤️ by Tanuj Bhatia**
