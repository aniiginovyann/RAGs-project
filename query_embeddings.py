import os
import pickle
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBEDDINGS_FILE = "embeddings.pkl"

with open(EMBEDDINGS_FILE, "rb") as f:
    embeddings_data = pickle.load(f)

def get_most_similar_chunks(query, embeddings_data, top_k=5):
    resp = client.embeddings.create(
        model="text-embedding-3-large",
        input=query
    )
    query_emb = np.array(resp.data[0].embedding).reshape(1, -1)

    doc_embs = np.array([d["embedding"] for d in embeddings_data])
    similarities = cosine_similarity(query_emb, doc_embs)[0]

    top_indices = similarities.argsort()[-top_k:][::-1]
    top_chunks = [embeddings_data[i]["text"] for i in top_indices]
    return top_chunks

def generate_prompt(query, chunks):
    context = "\n\n".join(chunks)
    prompt = f"""
                Խնդրում եմ պատասխանիր հարցին միայն տրված կոնտեքստի հիման վրա։ 
                Մի ավելացրու սեփական կարծիք կամ լրացուցիչ ինֆորմացիա։

                Կոնտեքստ՝
                {context}

                Հարց՝
                {query}

                Պատասխան՝
            """
    return prompt

def ask_gpt(query, chunks):
    prompt = generate_prompt(query, chunks)
    response = client.chat.completions.create(
        model="gpt-5.2", 
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    while True:
        query = input("Մուտքագրեք ձեր հարցը: ")
        if query.lower() == "exit":
            break
        top_chunks = get_most_similar_chunks(query, embeddings_data, top_k=5)
        answer = ask_gpt(query, top_chunks)
        print("\nՊատասխան:\n", answer)
        print("-" * 80)