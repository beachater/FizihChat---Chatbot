from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.vectorstores import Chroma
from document_loader import load_documents_into_database
import os
import argparse
import sys

# Initialize FastAPI app
app = FastAPI(title="RAG API", description="A Retrieval-Augmented Generation API using Hugging Face models.")

# Allow CORS from the frontend
origins = [
    "http://localhost:39992",  # Ensure this matches your frontend port
    "http://127.0.0.1:55974",  # Add the specific port if it's different
    "http://localhost:8080",   # If needed for local dev
    "http://127.0.0.1:36804"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow any origin for now
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Allow OPTIONS for preflight
    allow_headers=["Content-Type", "Authorization", "*"],  # Allow necessary headers
)


# Load Hugging Face settings
repo_id = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
embedding_model_name = "nomic-embed-text"
documents_path = "Research"
db = None
rag_chain = None

# Initialize the database and RAG pipeline during app startup
@app.on_event("startup")
async def startup_event():
    global db, rag_chain

    if not os.getenv('HUGGING_ACCESS_TOKEN'):
        raise EnvironmentError("Please set the HUGGING_ACCESS_TOKEN environment variable.")

    # Load documents into the database
    try:
        db = load_documents_into_database(embedding_model_name, documents_path)
    except FileNotFoundError as e:
        raise RuntimeError(f"Error loading documents: {e}")

    # Initialize the Hugging Face embeddings and LLM
    embeddings = HuggingFaceEmbeddings()
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        huggingfacehub_api_token=os.getenv('HUGGING_ACCESS_TOKEN'),
        temperature=0.8,
        top_k=50
    )

    # Define the prompt template
    template = """
    You are a helpful assistant. Use the following context to answer concisely.
    If you don't know the answer, just say so.

    Context: {context}
    Question: {question}
    Answer:
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # Create RAG chain
    rag_chain = (
        {"context": db.as_retriever(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

# Define API endpoints
@app.post("/ask")
async def ask_question(request: Request, question: dict):
    """
    Process a user's question using the RAG pipeline.
    """
    global rag_chain, db
    if not rag_chain:
        raise HTTPException(status_code=500, detail="RAG pipeline not initialized.")

    try:
        question_text = question.get("question")
        if not question_text:
            raise HTTPException(status_code=400, detail="No question provided.")

        # Log incoming request headers for debugging
        print(f"Request Headers: {dict(request.headers)}")

        response = rag_chain.invoke({"context": db, "question": question_text})
        
        return {"question": question_text, "answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

# CLI Entry point for the original functionality
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local LLM with RAG using Hugging Face.")
    parser.add_argument(
        "-m",
        "--model",
        default="mistralai/Mixtral-8x7B-Instruct-v0.1",
        help="The name of the Hugging Face model repo to use.",
    )
    parser.add_argument(
        "-e",
        "--embedding_model",
        default="nomic-embed-text",
        help="The name of the embedding model to use.",
    )
    parser.add_argument(
        "-p",
        "--path",
        default="Research",
        help="The path to the directory containing documents to load.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    if not os.getenv('HUGGING_ACCESS_TOKEN'):
        print("Please set the HUGGING_ACCESS_TOKEN environment variable.")
        sys.exit(1)
    
    # Start the FastAPI app
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
