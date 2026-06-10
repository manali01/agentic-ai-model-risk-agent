import ollama
import chromadb

# 1. Sample model docs
docs = [
    "Credit risk model using logistic regression. Features include income, credit score, utilization, and delinquency history.",
    "Marketing response model using XGBoost. It predicts likelihood of accepting a Sapphire Reserve direct mail offer.",
    "Model validation includes out-of-time testing, backtesting, stability testing, and challenger model comparison.",
    "Potential issue: feature leakage if future payment behavior is included in training data.",
    "Potential issue: weak bias testing across protected or proxy demographic segments."
]

# 2. Create local Chroma database
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="model_docs")

# 3. Convert docs into embeddings using Ollama
for i, doc in enumerate(docs):
    embedding = ollama.embeddings(
        model="nomic-embed-text",
        prompt=doc
    )["embedding"]

    collection.add(
        ids=[f"doc_{i}"],
        documents=[doc],
        embeddings=[embedding]
    )

print("Documents added to retriever.")

# 4. Retriever function
def retrieve_relevant_docs(query, top_k=2):
    query_embedding = ollama.embeddings(
        model="nomic-embed-text",
        prompt=query
    )["embedding"]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results["documents"][0]

# 5. Test prompt goes here
question = "What risks should I check for this credit risk model?"

retrieved_docs = retrieve_relevant_docs(question)

print("\nRetrieved documents:")
for doc in retrieved_docs:
    print("-", doc)

# 6. Send retrieved context to Ollama LLM
prompt = f"""
You are a model risk reviewer.

Question:
{question}

Use only this retrieved context:
{retrieved_docs}

Return:
1. Key risks
2. Testing needed
3. Challenge questions
"""

response = ollama.chat(
    model="llama3",
    messages=[{"role": "user", "content": prompt}]
)

print("\nLLM Answer:")
print(response["message"]["content"])