from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone as ppc
from pinecone import ServerlessSpec as sc
import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

class Chatbot:

    def __init__(self):
        # Load and split documents
        loader = TextLoader('./INTJ.txt')
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
        self.docs = text_splitter.split_documents(documents)

        # Set up embeddings
        self.embeddings = HuggingFaceEmbeddings()

        # Initialize Pinecone
        self.index_name = "langchain-demo"
        self.pc = ppc(api_key="125fd599-d12d-419f-83b7-0c1e9d1945fe")

        # Create or use an existing index
        if self.index_name not in self.pc.list_indexes():
            self.pc.create_index(
                name=self.index_name,
                dimension=2,  # Replace with the actual dimension
                metric="cosine", 
                spec=sc(cloud="aws", region="us-east-1")
            )

        # Use Pinecone index as retriever
        self.retriever = Pinecone.from_existing_index(index_name=self.index_name)

        # Set up the language model
        repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            huggingfacehub_api_token=os.getenv('HUGGING_ACCESS_TOKEN'),
            temperature=0.8,
            top_k=50
        )

        # Define the prompt template
        template = """
        You are a fortune teller. The human will ask you questions about their life.
        Use the following context to answer concisely within 2 sentences.
        If you don't know the answer, just say so.

        Context: {context}
        Question: {question}
        Answer:
        """
        self.prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        # Create retrieval-augmented generation (RAG) chain
        self.rag_chain = (
            {"context": self.retriever.as_retriever(), "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def get_response(self, question):
        return self.rag_chain.invoke({"context": self.docs, "question": question})


# Initialize chatbot and take input
bot = Chatbot()
user_input = input("Ask me anything: ")
result = bot.get_response(user_input)
print(result)
