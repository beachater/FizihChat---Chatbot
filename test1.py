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


def main(llm_model_name: str, embedding_model_name: str, documents_path: str) -> None:
    # Creating database from documents
    try:
        db = load_documents_into_database(embedding_model_name, documents_path)
    except FileNotFoundError as e:
        print(e)
        sys.exit()

    # Set up Hugging Face embeddings
    embeddings = HuggingFaceEmbeddings()

    # Initialize the Hugging Face model
    repo_id = 'mistralai/Mixtral-8x7B-Instruct-v0.1'  # Replace with your Hugging Face model repo ID
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
    If question is not in the context then say that you cannot help with that and ask for relevant questions about the Context's "Title".
    If the question is a statement of approval, acknowledgement, and/or gratitude then respond accordingly. 

    Context: {context}
    Question: {question}
    Answer:
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # Create retrieval-augmented generation (RAG) chain
    rag_chain = (
        {"context": db.as_retriever(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    while True:
        try:
            user_input = input("\n\nPlease enter your question (or type 'exit' to end): ")
            if user_input.lower() == "exit":
                break

            response = rag_chain.invoke({"context": db, "question": user_input})
            print(f"Answer: {response}")
        except KeyboardInterrupt:
            break


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
    main(args.model, args.embedding_model, args.path)
    # def main(llm_model_name: str, embedding_model_name: str, documents_path: str) -> None:
    try:
        db = load_documents_into_database(embedding_model_name, documents_path)
    except FileNotFoundError as e:
        print(f"Error loading documents: {e}")
        sys.exit(1)

    embeddings = HuggingFaceEmbeddings()

    repo_id = llm_model_name
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        huggingfacehub_api_token=os.getenv('HUGGING_ACCESS_TOKEN'),
        temperature=0.8,
        top_k=50
    )

    template = """
    You are a helpful assistant. Use the following context to answer concisely.
    If you don't know the answer, just say so.

    Context: {context}
    Question: {question}
    Answer:
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    rag_chain = (
        {"context": db.as_retriever(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    while True:
        try:
            user_input = input("\n\nPlease enter your question (or type 'exit' to end): ")
            if user_input.lower() == "exit":
                break

            response = rag_chain.invoke({"context": db, "question": user_input})
            print(f"Answer: {response}")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

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
    main(args.model, args.embedding_model, args.path)