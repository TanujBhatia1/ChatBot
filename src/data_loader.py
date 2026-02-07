from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader

# Loaders for different file types
class Loaders:
    PDF = PyMuPDFLoader
    TXT = TextLoader
    CSV = CSVLoader
    DOCX = Docx2txtLoader
    XLSX = UnstructuredExcelLoader
    JSON = JSONLoader


    @classmethod
    def load_all_documents(cls, data_dir: str):
        """
        Load all documents from the specified directory, handling various file types.
        Supported file types include PDF, TXT, CSV, DOCX, XLSX, and JSON.
        """
    
        pdf_dir = Path(data_dir).resolve()
        print(f"[DEBUG] Loading documents from directory: {pdf_dir}")
        
        pdf_files = list(pdf_dir.glob("**/*.pdf"))

        all_documents = []
        print(f"[DEBUG] Found {len(pdf_files)} PDF files in directory {data_dir}")
        for pdf_file in pdf_files:
            print(f"Processing file: {pdf_file}")
            try:
                loader = PyMuPDFLoader(str(pdf_file))
                documents = loader.load()

                # Add source information to metadata
                for doc in documents:
                    doc.metadata["source"] = pdf_file.name
                    doc.metadata["file_type"] = "pdf"
                all_documents.extend(documents)
                print(f"[DEBUG] Loaded {len(documents)} documents from {pdf_file}")
            except Exception as e:
                print(f"[ERROR] Failed to load PDF {pdf_file}: {e}")
        
        return all_documents


    # Load Text files
    @classmethod
    def load_text_files(cls, data_dir: str):
        text_files = list(Path(data_dir).resolve().glob("**/*.txt"))
        print(f"[DEBUG] Found {len(text_files)} TXT files in directory {data_dir}")
        all_documents = []
        for text_file in text_files:    
            print(f"Processing file: {text_file}")
            try:
                loader = TextLoader(str(text_file))
                documents = loader.load()

                # Add source information to metadata
                for doc in documents:
                    doc.metadata["source"] = text_file.name
                    doc.metadata["file_type"] = "txt"
                all_documents.extend(documents)
                print(f"[DEBUG] Loaded {len(documents)} documents from {text_file}")
            except Exception as e:
                print(f"[ERROR] Failed to load TXT {text_file}: {e}")
                
        return all_documents

    # Load CSV files
    @classmethod
    def load_csv_files(cls, data_dir: str):
        csv_files = list(Path(data_dir).resolve().glob("**/*.csv"))
        print(f"[DEBUG] Found {len(csv_files)} CSV files in directory {data_dir}")
        all_documents = []
        for csv_file in csv_files:
            print(f"Processing file: {csv_file}")
            try:
                loader = CSVLoader(str(csv_file))
                documents = loader.load()

                # Add source information to metadata
                for doc in documents:
                    doc.metadata["source"] = csv_file.name
                    doc.metadata["file_type"] = "csv"
                all_documents.extend(documents)
                print(f"[DEBUG] Loaded {len(documents)} documents from {csv_file}")
            except Exception as e:
                print(f"[ERROR] Failed to load CSV {csv_file}: {e}")
        return all_documents
    
    # Load Docx files
    @classmethod
    def load_docx_files(cls, data_dir: str):
        docx_files = list(Path(data_dir).resolve().glob("**/*.docx"))
        print(f"[DEBUG] Found {len(docx_files)} DOCX files in directory {data_dir}")
        all_documents = []
        for docx_file in docx_files:
            print(f"Processing file: {docx_file}")
            try:
                loader = Docx2txtLoader(str(docx_file))
                documents = loader.load()

                # Add source information to metadata
                for doc in documents:
                    doc.metadata["source"] = docx_file.name
                    doc.metadata["file_type"] = "docx"
                all_documents.extend(documents)
                print(f"[DEBUG] Loaded {len(documents)} documents from {docx_file}")
            except Exception as e:
                print(f"[ERROR] Failed to load DOCX {docx_file}: {e}")
        return all_documents
    
    
loaders = Loaders()