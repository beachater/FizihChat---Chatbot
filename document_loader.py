from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
import os
from typing import List
from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

from tqdm import tqdm

def load_documents_into_database(model_name: str, documents_path: str) -> Chroma:
    """
    Loads documents into a persistent Chroma database. If the database exists,
    it loads it directly; otherwise, it processes the documents and saves the embeddings.

    Args:
        model_name (str): The embedding model name.
        documents_path (str): Path to the directory containing the documents.

    Returns:
        Chroma: The Chroma database with loaded documents.
    """
    db_path = "./chroma_db"  # Directory to store the Chroma database
    
    # Check if the database already exists
    if os.path.exists(db_path):
        print("Database exists. Loading embeddings directly...")
        db = Chroma(persist_directory=db_path, embedding_function=OllamaEmbeddings(model=model_name))
        return db

    print("Step 1: Loading raw documents")
    raw_documents = load_documents(documents_path)
    print(f"Loaded {len(raw_documents)} documents from '{documents_path}'.")

    print("Step 2: Splitting documents into manageable chunks")
    documents = TEXT_SPLITTER.split_documents(raw_documents)
    print(f"Split into {len(documents)} chunks.")

    print("Step 3: Creating embeddings and loading into Chroma")
    embeddings = OllamaEmbeddings(model=model_name)

    # Process documents with progress bar
    document_texts = [doc.page_content for doc in documents]
    document_metadatas = [doc.metadata for doc in documents]
    
    processed_texts = []
    processed_metadatas = []

    with tqdm(total=len(document_texts), desc="Processing documents", unit="doc") as pbar:
        for text, metadata in zip(document_texts, document_metadatas):
            processed_texts.append(text)
            processed_metadatas.append(metadata)

    # Create the Chroma database from the processed texts and metadata
    db = Chroma.from_texts(
        processed_texts,
        embedding=embeddings,
        metadatas=processed_metadatas,
        persist_directory=db_path,  # Save the database to this directory
    )
    
    db.persist()  # Persist the database to disk
    print("Completed: Embeddings created and documents loaded into Chroma.")
    return db


def load_documents(path: str) -> List[Document]:
    """
    Loads documents from the specified directory path.

    This function supports loading of PDF, Markdown, and HTML documents by utilizing
    different loaders for each file type. It checks if the provided path exists and
    raises a FileNotFoundError if it does not. It then iterates over the supported
    file types and uses the corresponding loader to load the documents into a list.

    Args:
        path (str): The path to the directory containing documents to load.

    Returns:
        List[Document]: A list of loaded documents.

    Raises:
        FileNotFoundError: If the specified path does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The specified path does not exist: {path}")

    loaders = {
        ".pdf": DirectoryLoader(
            path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
            use_multithreading=True,
        ),
        ".md": DirectoryLoader(
            path,
            glob="**/*.md",
            loader_cls=TextLoader,
            show_progress=True,
        ),
    }

    docs = []
    for file_type, loader in loaders.items():
        print(f"Loading {file_type} files")
        docs.extend(loader.load())
    return docs
