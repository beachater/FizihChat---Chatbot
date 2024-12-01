from flask import Flask, request, jsonify
from langchain_huggingface import HuggingFaceEndpoint
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os



# Initialize chatbot components
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
huggingface_token = os.getenv("HUGGING_ACCESS_TOKEN")

# Set up LLM
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    huggingfacehub_api_token=huggingface_token,
    temperature=0.8,
    top_k=50
)

# Set up prompt template
template = """
You are a helpful assistant. Use the following context to answer concisely.
If you don't know the answer, just say so.

Context: {context}
Question: {question}
Answer:
"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# Set up embeddings and database
embeddings = HuggingFaceEmbeddings()
db = Chroma()  # Ensure your document loader creates a compatible database
rag_chain = (
    {"context": db.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

import os
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400
    response = {"answer": f"This is a mock response to: {question}"}
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)