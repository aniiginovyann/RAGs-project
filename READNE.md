# 📚 RAGs System Project

This project implements a **Retrieval-Augmented Generation (RAG) system** using OpenAI embeddings and GPT models.  
It allows querying PDF documents and generates answers based on the most relevant document segments.

---

## ✨ Features

- 📄 Load PDF documents from a directory.
- ✂️ Split documents into smaller chunks for embeddings.
- ⚡ Generate embeddings for each chunk using `text-embedding-3-large`.
- 💾 Store embeddings locally in a pickle file.
- 🔍 Retrieve top relevant chunks using cosine similarity.
- 🤖 Generate GPT-based answers strictly from the retrieved context.
- 🖥️ Interactive console query interface.

---

## 🛠 Installation

1. Clone the repository:

'''bash
git clone <repository_url>
cd <repository_directory>

2. Create a virtual environment and activate it:
python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows

3. Install dependencies:
pip install -r requirements.txt

4. Setup environment variables:
Create a .env file with your OpenAI API key:
OPENAI_API_KEY=your_openai_api_key_here

## 🚀 Usage
Step 1: Generate embeddings
python generate_embeddings.py

Loads PDFs from Docs/
Splits them into chunks
Creates embeddings and saves to embeddings.pkl

Step 2: Ask questions
python query_rag.py
Enter your query when prompted
Retrieves the most relevant chunks
GPT-5.2 generates an answer based on context
Type exit to quit


## 🗂️ Project Structure

rag-project/
├─ Docs/   
├─ generate_embeddings.py
├─ query_rag.py          
├─ embeddings.pkl       
├─ requirements.txt    
├─ .env                  
└─ README.md            
└─ README.md             
