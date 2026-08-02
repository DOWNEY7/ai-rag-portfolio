import os
# [Windows Hotfix] Ensure Python in Conda finds the correct sqlite3.dll
if os.name == 'nt':
    conda_base = r"C:\Users\MINNAS\anaconda3\Library\bin"
    if os.path.exists(conda_base):
        os.environ["PATH"] = conda_base + ";" + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(conda_base)
        except AttributeError:
            pass

from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

CORPUS_DIR = "./corpus"
CHROMA_PERSIST_DIR = "./chroma_db"

def load_documents(corpus_dir: str):
    """
    Iterates through the corpus directory and loads .md and .pdf files.
    """
    documents = []
    corpus_path = Path(corpus_dir)
    
    if not corpus_path.exists():
        print(f"Error: Corpus directory '{corpus_dir}' does not exist.")
        return documents

    print(f"Scanning '{corpus_dir}' for documents...")
    for root, _, files in os.walk(corpus_dir):
        for file in files:
            file_path = Path(root) / file
            
            try:
                if file.endswith(".md"):
                    loader = UnstructuredMarkdownLoader(str(file_path))
                    documents.extend(loader.load())
                elif file.endswith(".pdf"):
                    loader = PyPDFLoader(str(file_path))
                    documents.extend(loader.load())
                # Ignore other file types silently
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
    return documents

def chunk_documents(documents):
    """
    Splits the loaded documents into chunks using the tiktoken encoder.
    """
    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=700,
        chunk_overlap=100
    )
    return text_splitter.split_documents(documents)

def build_vector_store(chunks, persist_directory: str):
    """
    Builds and persists a Chroma vector store from document chunks via OpenRouter OpenAIEmbeddings.
    """
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY or OPENAI_API_KEY environment variable is missing. Please add it to your .env file.")
        
    print("Initializing embeddings via OpenRouter...")
    embeddings = OpenAIEmbeddings(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        model="text-embedding-3-small"
    )
    
    print(f"Building vector store at '{persist_directory}'...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    return vector_store

def main():
    print("--- Starting Data Ingestion Pipeline ---")
    
    # 1. Load documents
    documents = load_documents(CORPUS_DIR)
    
    if not documents:
        print("No valid documents found. Exiting.")
        return
        
    print(f"Total documents loaded: {len(documents)}")
    
    # 2. Chunk text
    chunks = chunk_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    
    # 3. Build and save database
    try:
        build_vector_store(chunks, CHROMA_PERSIST_DIR)
        print("✅ Vector database successfully created and saved!")
    except Exception as e:
        print(f"❌ Failed to build vector database: {e}")

if __name__ == "__main__":
    main()
