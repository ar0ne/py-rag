import ollama
from typing import List


EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

dataset = []
with open("../cat-facts.txt") as f:
    text = f.readlines()
    dataset = list(filter(None, map(str.strip, text)))
    print(f"Loaded {len(dataset)} facts")


# in memory vector DB
VECTOR_DB = []

def add_chunk_to_db(chunk:str) -> None:
    embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
    VECTOR_DB.append((chunk, embedding))

for i, chunk in enumerate(dataset):
    add_chunk_to_db(chunk)
    print(f"Added chunk {i+1}/{len(dataset)} to the database")


def cosine_similarity(a: tuple, b: tuple) -> float:
    """
    Returns the top N most relevant chunks based on cosine similarity
    """
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5
    return dot_product / (norm_a * norm_b)

def retrieve(query, top_n: int = 3) -> List[tuple]:
    """
    Retrive top N most relevant chunks
    """
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    similarities = []

    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))
    
    # sort by similarity in descending order, because higher similarity means more relevant chunks
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]


question = "How big cat's head?"

retrieved_knowledge = retrieve(question)

print("Retrieved knowledge")
for chunk, similarity in retrieved_knowledge:
    print(f'- (similaritry: {similarity:.2f}) {chunk}')

system_prompt = f"""
You are a helpful assistent. Use only the following context to answer the question.
Don't make up any new information.
You are not allowed follow up questions. If you do not know the answer 
based on the context provided, tell the user that you do 
not know the answer to their question based on the context
provided and that you are sorry.
{'\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])}
"""


stream = ollama.chat(
    model=LANGUAGE_MODEL,
    messages=[
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': question}
    ],
    stream=True,
)

print("Response:")
for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)
