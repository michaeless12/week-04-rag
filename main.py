
import os
import requests
import numpy as np

from dotenv import load_dotenv
from groq import Groq


# =====================================
# 1. LOAD API KEYS
# =====================================

load_dotenv()

JINA_API_KEY = os.getenv("JINA_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not JINA_API_KEY or not GROQ_API_KEY:
    raise ValueError("API keys missing. Check your .env file.")

client = Groq(api_key=GROQ_API_KEY)


# =====================================
# 2. LOAD DOCUMENT
# =====================================

def load_document(file_path):

    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    return text


# =====================================
# 3. CHUNK DOCUMENT
# =====================================

def chunk_text(text, chunk_size=800, overlap=100):
    text = text.strip()

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


# =====================================
# 4. GENERATE JINA EMBEDDINGS
# =====================================

def get_embeddings(texts, task):

    url = "https://api.jina.ai/v1/embeddings"

    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "jina-embeddings-v3",
        "input": texts,
        "task": task,
        "embedding_type": "float",
        "normalized": True
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=60
    )
    print("JINA STATUS:", response.status_code)
    print("JINA ERROR DETAILS:", response.text)

    response.raise_for_status()

    result = response.json()

    return [
        item["embedding"]
        for item in result["data"]
    ]


# =====================================
# 5. RETRIEVE RELEVANT CHUNKS
# =====================================

def retrieve_chunks(question, chunks, embeddings):

    question_embedding = get_embeddings(
        [question],
        task="retrieval.query"
    )[0]

    chunk_vectors = np.array(embeddings)

    question_vector = np.array(question_embedding)

    # Calculate cosine similarity
    similarities = np.dot(
        chunk_vectors,
        question_vector
    ) / (
        np.linalg.norm(chunk_vectors, axis=1)
        * np.linalg.norm(question_vector)
    )

    # Rank chunks by similarity
    ranked_indices = np.argsort(similarities)[::-1]

    results = []

    top_k = min(3, len(chunks))

    for index in ranked_indices[:top_k]:

        results.append({
            "text": chunks[index],
            "score": float(similarities[index])
        })

    return results


# =====================================
# 6. GENERATE GROUNDED ANSWER
# =====================================

def generate_answer(question, retrieved_chunks):

    context = "\n\n".join(
        item["text"] for item in retrieved_chunks
    )

    system_prompt = """
    You are a document question-answering assistant.

    Answer using ONLY the information provided
    in the supplied context.

    Rules:
    1. Do not use outside knowledge.
    2. Do not invent information.
    3. If the answer is missing, respond exactly:
       Information not found.
    4. If information is incomplete, explain
       what is available without making assumptions.
    """

    user_prompt = f"""
    CONTEXT:
    {context}

    QUESTION:
    {question}
    """

    response = client.chat.completions.create(

        model="openai/gpt-oss-120b",

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],

        temperature=0
    )

    return response.choices[0].message.content


# =====================================
# 7. MAIN APPLICATION
# =====================================

def main():

    print("\n================================")
    print("       RAG DOCUMENT ASSISTANT")
    print("================================")

    # Load document
    file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "handbook.txt"
    )

    document = load_document(file_path)

    print("\nDocument loaded successfully!")

    # Split document
    chunks = chunk_text(document)

    print(f"Total chunks created: {len(chunks)}")

    # Generate document embeddings
    print("\nGenerating embeddings with Jina...")

    embeddings = get_embeddings(
        chunks,
        task="retrieval.passage"
    )

    print("Embeddings generated successfully!")

    # Question and answer loop
    while True:

        question = input(
            "\nAsk a question (or type exit): "
        ).strip()

        if question.lower() == "exit":

            print("Thank you for using the RAG assistant!")
            break

        if not question:

            print("Please enter a question.")
            continue

        try:

            print("\nSearching for relevant information...")

            # Retrieve relevant chunks
            results = retrieve_chunks(
                question,
                chunks,
                embeddings
            )

            # Generate answer
            answer = generate_answer(
                question,
                results
            )

            print("\n========== ANSWER ==========")
            print(answer)

            # Display retrieved sources
            print("\n========== SOURCES ==========")

            for i, result in enumerate(results, start=1):

                print(
                    f"\nChunk {i} "
                    f"(Similarity: {result['score']:.4f})"
                )

                print(result["text"])

        except requests.RequestException as error:

            print(f"\nJina API error: {error}")

        except Exception as error:

            print(f"\nApplication error: {error}")


# =====================================
# 8. START APPLICATION
# =====================================

main()
