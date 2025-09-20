from dotenv import load_dotenv
from src.helper import load_pdf_files, filter_to_minimal_docs, chunking, download_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# Load and process documents, Chunking and downloading embeddings
extracted_docs = load_pdf_files("data/")
minimal_docs = filter_to_minimal_docs(extracted_docs)
text_chunk = chunking(minimal_docs)
embedding = download_embeddings()


pc = Pinecone()

index_name = "ai-doctor"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,  # Dimension of the embedding model
        metric="cosine",  # Similarity metric
        # pod_type="p1",  # Pod type
        spec=ServerlessSpec(cloud="aws", region="us-east-1")  # Serverless configuration
    )

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunk,
    embedding=embedding,
    index_name=index_name,
)