from dotenv import load_dotenv
import os
import pickle
from openai import OpenAI
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()  

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set!")

client = OpenAI(api_key=api_key)
DOCS_PATH = "Docs"
EMBEDDINGS_FILE = "embeddings.pkl"

def load_documents(docs_path=DOCS_PATH):
    pdf_loader = DirectoryLoader(
        path=docs_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = pdf_loader.load()
    return documents

def split_documents(documents, chunk_size=1500, chunk_overlap=300):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)
    return chunks

def create_embeddings(chunks):
    embeddings = []
    for doc in chunks:
        resp = client.embeddings.create(
            model="text-embedding-3-large", 
            input=doc.page_content
        )
        embeddings.append({
            "text": doc.page_content,
            "embedding": resp.data[0].embedding
        })

        with open(EMBEDDINGS_FILE, "wb") as f:
            pickle.dump(embeddings, f)

def load_embeddings():
    with open(EMBEDDINGS_FILE, "rb") as f:
        embeddings = pickle.load(f)
    return embeddings

if __name__ == "__main__":
    documents = load_documents()
    chunks = split_documents(documents)
    create_embeddings(chunks)

    embeddings = load_embeddings()
