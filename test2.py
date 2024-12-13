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
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fpdf import FPDF
from psycopg2 import sql


user_session = {}

# Directory to store uploaded files
UPLOAD_DIR: str = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)  # Create the directory if it doesn't exist

infodirectory = "business_initial_information"
if not os.path.exists(infodirectory):
    os.makedirs(infodirectory)


# Pydantic model for the input data
class BusinessInfo(BaseModel):
    companyName: str
    description: str
    scope: str
    limitations: str


class UserInfoUpdate(BaseModel):
    email: str
    username: str

    
    
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
documents_path = "business_initial_information"
db_rag = None
rag_chain = None

## Utility function for initializing RAG pipeline
def initialize_rag_pipeline():
    global db_rag, rag_chain

    # Check if the Hugging Face access token is set
    if not os.getenv('HUGGING_ACCESS_TOKEN'):
        raise EnvironmentError("Please set the HUGGING_ACCESS_TOKEN environment variable.")

    # Load documents into the database
    try:
        db_rag = load_documents_into_database("nomic-embed-text", "business_initial_information")
    except FileNotFoundError as e:
        raise RuntimeError(f"Error loading documents: {e}")

    # Initialize the Hugging Face embeddings and LLM
    embeddings = HuggingFaceEmbeddings()
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
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
    return {"message": "RAG pipeline initialized successfully."}

@app.post("/initialize-pipeline")
async def initialize_pipeline():
    """
    Initialize the RAG pipeline without saving new data.
    """
    try:
        result = initialize_rag_pipeline()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/save-business-info")
async def save_business_info(info: BusinessInfo):
    """
    Save business information and initialize RAG pipeline if not already initialized.
    """
    global db_rag, rag_chain

    # Ensure the user is logged in by checking the user_session
    if 'user_id' not in user_session:
        raise HTTPException(status_code=401, detail="User not logged in.")
    
    user_id = user_session['user_id']

    try:
        # Save the business information in a .pdf file
        file_path = os.path.join(infodirectory, f"{info.companyName.replace(' ', '_')}.pdf")

        # Create a PDF instance
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # Set font
        pdf.set_font("Arial", size=12)

        # Add content to the PDF
        pdf.cell(200, 10, txt=f"Company Name: {info.companyName}", ln=True)
        pdf.cell(200, 10, txt=f"Description: {info.description}", ln=True)
        pdf.cell(200, 10, txt=f"Scope: {info.scope}", ln=True)
        pdf.cell(200, 10, txt=f"Limitations: {info.limitations}", ln=True)

        # Save the PDF
        pdf.output(file_path)
        
        conn = db
        cursor = conn.cursor()

        # Insert business information into the database with user_id
        insert_query = sql.SQL("""
            INSERT INTO business_info (company_name, description, scope, limitations, user_id)
            VALUES (%s, %s, %s, %s, %s)
        """)
        cursor.execute(insert_query, (info.companyName, info.description, info.scope, info.limitations, user_id))

        # Commit the transaction
        conn.commit()

        # Close the cursor and connection
        cursor.close()
        conn.close()

        # Initialize RAG pipeline if not already done
        if not db_rag or not rag_chain:
            initialize_rag_pipeline()

        return {"message": "Business information saved and RAG pipeline initialized successfully."}
    except Exception as e:
        return {"error": str(e)}
    
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
        
        print("Upload Successfull ------------------")

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
        
        user_session['user_id'] = user['id']
        print(user_session['user_id'])
        
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

@app.get("/get-user-settings")
async def get_user_settings():
    if 'user_id' not in user_session:
        raise HTTPException(status_code=401, detail="User not logged in.")
    
    user_id = user_session['user_id']

    try:
        with psycopg2.connect(
            database="fizihchat_db",
            user="myuser",
            password="mypassword",
            host="localhost",
        ) as connection:
            with connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("SELECT id, email FROM users WHERE id = %s", (user_id,))
                user = cursor.fetchone()
                if not user:
                    raise HTTPException(status_code=404, detail="User not found.")
                return {"email": user["email"]}
    except psycopg2.Error as err:
        raise HTTPException(status_code=500, detail=f"Database error: {err}")

@app.get("/get-company-info")
async def get_company_info():
    if 'user_id' not in user_session:
        raise HTTPException(status_code=401, detail="User not logged in.")
    
    user_id = user_session['user_id']

    try:
        with psycopg2.connect(
            database="fizihchat_db",
            user="myuser",
            password="mypassword",
            host="localhost",
        ) as connection:
            with connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT company_name, description, scope, limitations 
                    FROM business_info 
                    WHERE user_id = %s
                """, (user_id,))
                company = cursor.fetchone()
                if not company:
                    raise HTTPException(status_code=404, detail="Company not found.")
                return company
    except psycopg2.Error as err:
        raise HTTPException(status_code=500, detail=f"Database error: {err}")


@app.post("/update-user-info")
async def update_user_info(info: UserInfoUpdate):
    """
    Update user information (email and username).
    """
    # Ensure the user is logged in
    if 'user_id' not in user_session:
        raise HTTPException(status_code=401, detail="User not logged in.")
    
    user_id = user_session['user_id']
    
    cursor = db.cursor()
    try:
        # Update the user info (email, username) in the database
        update_query = """
            UPDATE users 
            SET email = %s, username = %s 
            WHERE id = %s
        """
        cursor.execute(update_query, (info.email, info.username, user_id))
        db.commit()
        
        return {"message": "User information updated successfully"}
    except psycopg2.Error as err:
        raise HTTPException(status_code=500, detail=f"Database error: {err}")
    finally:
        cursor.close()

class ChangePassword(BaseModel):
    old_password: str
    new_password: str

@app.post("/change-password")
async def change_password(change_info: ChangePassword):
    """
    Change the user's password after verifying the old password.
    """
    # Ensure the user is logged in
    if 'user_id' not in user_session:
        raise HTTPException(status_code=401, detail="User not logged in.")
    
    user_id = user_session['user_id']
    
    cursor = db.cursor(cursor_factory=RealDictCursor)
    try:
        cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
        user = cursor.fetchone()

        if not user:
            raise HTTPException(status_code=404, detail="User not found.")
        
        # Verify if the old password is correct
        if not check_password_hash(user['password'], change_info.old_password):
            raise HTTPException(status_code=401, detail="Old password is incorrect.")

        # Hash the new password and update it
        hashed_new_password = generate_password_hash(change_info.new_password)
        
        cursor.execute("UPDATE users SET password = %s WHERE id = %s", (hashed_new_password, user_id))
        db.commit()

        return {"message": "Password updated successfully"}
    except psycopg2.Error as err:
        raise HTTPException(status_code=500, detail=f"Database error: {err}")
    finally:
        cursor.close()

        
        
@app.post("/update-business-info")
async def update_business_info(
    user_id: int = Form(...),
    companyName: str = Form(...),
    description: str = Form(...),
    scope: str = Form(...),
    limitations: str = Form(...)
):
    """
    Update business information for the logged-in user.
    """
    # Ensure the user is logged in
    if 'user_id' not in user_session or user_session['user_id'] != user_id:
        raise HTTPException(status_code=401, detail="User not logged in or unauthorized.")

    cursor = db.cursor()
    try:
        # Update the business information in the database
        update_query = sql.SQL("""
            UPDATE business_info 
            SET company_name = %s, description = %s, scope = %s, limitations = %s
            WHERE user_id = %s
        """)
        cursor.execute(update_query, (companyName, description, scope, limitations, user_id))
        
        db.commit()

        # Update the PDF file as well
        file_path = os.path.join(infodirectory, f"{companyName.replace(' ', '_')}.pdf")
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Company Name: {companyName}", ln=True)
        pdf.cell(200, 10, txt=f"Description: {description}", ln=True)
        pdf.cell(200, 10, txt=f"Scope: {scope}", ln=True)
        pdf.cell(200, 10, txt=f"Limitations: {limitations}", ln=True)

        pdf.output(file_path)

        return {"message": "Business information updated successfully"}
    except psycopg2.Error as err:
        raise HTTPException(status_code=500, detail=f"Database error: {err}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
    finally:
        cursor.close()


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
