from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings

# Extract text from PDF files
def load_pdf_files(filepath):
    loader = DirectoryLoader(filepath, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents



# Remove metadata and filter documents to keep only minimal necessary information
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs = []
    for doc in docs:
        src = doc.metadata.get("source")
        if len(doc.page_content) > 100:  # Example criterion: keep documents with more than 100 characters
            minimal_docs.append(
                Document(page_content=doc.page_content, metadata={"source": src})
                )
    return minimal_docs


# split the documents into smaller chunks
def chunking(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )

    text_chunk = text_splitter.split_documents(minimal_docs)
    return text_chunk

# Download pre-trained embeddings from HuggingFace
def download_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings