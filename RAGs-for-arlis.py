import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import logging
import sys
from openai import OpenAI

sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logging_history.log")
        ]
)

logger = logging.getLogger(__name__)

def load_documents(docs_path="Docs"):
    logger.info(f"Loading documents from {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} doesn't exist. Please create it and add your files.")
    
    pdf_loader = DirectoryLoader(
        path=docs_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = []
    documents.extend(pdf_loader.load())

    if len(documents) == 0:
        raise FileNotFoundError(f"No .pdf files found in {docs_path}. Please add your documents.")
    
    return documents
    
def split_documents(documents, chunk_size=3000, chunk_overlap=200):
    logger.info("Splitting documents into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)
    
    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    logger.info("Creating embeddings and storing in ChromaDB...")

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    logger.info("Creating vector store...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    logger.info("Finished creating vector store.")
    logger.info(f"Vector store created and saved to {persist_directory}.")

    return vectorstore

def ask_gpt(query, docs):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""
        Խնդրում եմ պատասխանիր հարցին միայն տրված կոնտեքստի հիման վրա։ 
        Մի ավելացրու սեփական կարծիք կամ լրացուցիչ ինֆորմացիա։

        Կոնտեքստ՝
        {context}

        Հարց՝
        {query}

        Պատասխան՝
    """
    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages = [
            {"role": "user", "content": prompt}
        ],
        temperature = 0
    )

    return response.choices[0].message.content



def main():
    documents = load_documents(docs_path="Docs")
    chunks = split_documents(documents)
    vectorstore = create_vector_store(chunks)

    query = input("Մուտքագրեք ձեր հարցը ՀՀ օրենսդրության վերաբերյալ: ")
    docs = vectorstore.similarity_search(query, k=5)
    answer = ask_gpt(query, docs)

    print(answer)
if __name__ == "__main__":
    main()
