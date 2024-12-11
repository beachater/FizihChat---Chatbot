from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from werkzeug.security import generate_password_hash
from werkzeug.security import check_password_hash
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
import shutil
import psycopg2
from psycopg2.extras import RealDictCursor

# Directory to store uploaded files
UPLOAD_DIR: str = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)  # Create the directory if it doesn't exist

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

# Database connection setup for PostgreSQL
db = psycopg2.connect(
    host="localhost",
    user="myuser",  # Replace with your PostgreSQL user
    password="mypassword",  # Replace with your PostgreSQL password
    database="fizihchat_db"     # Replace with your database name
)

# Load Hugging Face settings
repo_id = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
embedding_model_name = "nomic-embed-text"
documents_path = "Research"
db_rag = None
rag_chain = None

# Initialize the database and RAG pipeline during app startup
@app.on_event("startup")
async def startup_event():
    global db_rag, rag_chain

    if not os.getenv('HUGGING_ACCESS_TOKEN'):
        raise EnvironmentError("Please set the HUGGING_ACCESS_TOKEN environment variable.")

    # Load documents into the database
    try:
        db_rag = load_documents_into_database(embedding_model_name, documents_path)
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
        {"context": db_rag.as_retriever(), "question": RunnablePassthrough()}
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
    global rag_chain, db_rag
    if not rag_chain:
        raise HTTPException(status_code=500, detail="RAG pipeline not initialized.")

    try:
        question_text = question.get("question")
        if not question_text:
            raise HTTPException(status_code=400, detail="No question provided.")

        # Log incoming request headers for debugging
        print(f"Request Headers: {dict(request.headers)}")

        response = rag_chain.invoke({"context": db_rag, "question": question_text})
        
        return {"question": question_text, "answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Endpoint to upload files and store them for processing.
    """
    try:
        file_path = os.path.join(UPLOAD_DIR, str(file.filename))
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract text and add to Chroma database
        text = extract_text_from_file(file_path)
        add_text_to_chromadb(text)

        return {"filename": file.filename, "status": "uploaded successfully"}
    except Exception as e:
        print(f"Error during file upload: {e}")  # Log the error to the terminal
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/register")
async def register_user(email: str = Form(...), password: str = Form(...)):
    """
    Register a new user with email and password.
    """
    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password are required.")

    # Hash the password for security
    hashed_password = generate_password_hash(password)
    cursor = db.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (email, password) VALUES (%s, %s)",
            (email, hashed_password)
        )
        db.commit()
        return {"message": "User registered successfully"}
    except psycopg2.Error as err:
        raise HTTPException(status_code=500, detail=f"Database error: {err}")
    finally:
        cursor.close()


@app.post("/login")
async def login_user(email: str = Form(...), password: str = Form(...)):
    """
    Authenticate a user by verifying their email and password.
    """
    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password are required.")

    cursor = db.cursor(cursor_factory=RealDictCursor)
    try:
        # Fetch the user by email
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()

        if not user:
            raise HTTPException(status_code=404, detail="User not found.")
        
        # Check if the password matches
        if not check_password_hash(user['password'], password):
            raise HTTPException(status_code=401, detail="Invalid password.")
        
        return {"message": "Login successful", "user_id": user['id']}
    except psycopg2.Error as err:
        raise HTTPException(status_code=500, detail=f"Database error: {err}")
    finally:
        cursor.close()

@app.get("/uploads")
async def list_uploaded_files():
    """
    Return a list of uploaded files in the UPLOAD_DIR.
    """
    try:
        # Ensure the upload directory exists
        if not os.path.exists(UPLOAD_DIR):
            return JSONResponse(content={"files": []})  # No files uploaded yet

        # List files in the upload directory
        files = os.listdir(UPLOAD_DIR)
        return JSONResponse(content={"files": files})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


def extract_text_from_file(file_path):
    """
    Extract text from PDF or TXT files.
    """
    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif file_path.endswith(".pdf"):
        # Use a library like PyPDF2 or pdfminer to extract text
        from PyPDF2 import PdfReader
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() for page in reader.pages)
    else:
        raise ValueError("Unsupported file format")
    
def add_text_to_chromadb(text):
    """
    Add extracted text to Chroma database.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)

    global db_rag
    if not db_rag:
        raise RuntimeError("Database not initialized")
    db_rag.add_texts(texts=chunks)

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
