import ollama
import chromadb

# -----------------------------
# TOOL 1: Risk checklist
# -----------------------------
def get_model_risk_rules():
    return [
        "Data leakage",
        "Overfitting",
        "Bias in training data",
        "Weak independent validation",
        "Feature instability",
    ]


# -----------------------------
# TOOL 2: Retriever
# This searches your small knowledge base
# -----------------------------
docs = [
    "Credit risk model using logistic regression with income and credit score features.",
    "Marketing model using XGBoost for customer response prediction.",
    "Validation includes out-of-time testing and backtesting.",
    "Risk: feature leakage if future data is used.",
    "Risk: bias if demographic segments are not tested.",
]

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="model_docs_react")

# Add docs to vector database
for i, doc in enumerate(docs):
    embedding = ollama.embeddings(
        model="nomic-embed-text",
        prompt=doc
    )["embedding"]

    collection.upsert(
        ids=[f"doc_{i}"],
        documents=[doc],
        embeddings=[embedding]
    )


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


# -----------------------------
# User question
# -----------------------------
question = "What risks should I check for a credit risk model?"


# -----------------------------
# Agent instructions
# This tells LLM HOW to behave
# -----------------------------
messages = [
    {
        "role": "system",
        "content": """
You are a model risk reviewer.

You can use these tools:

1. get_model_risk_rules
Use this when you need standard model risk checklist.

2. retrieve_relevant_docs
Use this when you need relevant past model documentation or issues.

Use this exact format:

Thought: explain what you need to do
Action: tool_name

Allowed actions:
Action: get_model_risk_rules
Action: retrieve_relevant_docs

After you receive an Observation, either use another Action or provide:

Final Answer:
Risk:
Why it matters:
Testing needed:

Do not invent tool names.
Keep final answer short.
"""
    },
    {
        "role": "user",
        "content": question
    }
]


# -----------------------------
# Agent loop
# -----------------------------
for step in range(5):

    print(f"\n========== STEP {step + 1} ==========")

    response = ollama.chat(
        model="llama3",
        messages=messages,
    )

    reply = response["message"]["content"]

    print("\nLLM RESPONSE:")
    print(reply)

    # Save LLM response to memory
    messages.append({
        "role": "assistant",
        "content": reply
    })

    # -----------------------------
    # If LLM chooses risk rules tool
    # -----------------------------
    if "Action: get_model_risk_rules" in reply:

        print("\nPYTHON ACTION:")
        print("Calling get_model_risk_rules()")

        tool_result = get_model_risk_rules()

        print("\nOBSERVATION:")
        print(tool_result)

        messages.append({
            "role": "user",
            "content": f"Observation: {tool_result}"
        })

    # -----------------------------
    # If LLM chooses retriever tool
    # -----------------------------
    elif "Action: retrieve_relevant_docs" in reply:

        print("\nPYTHON ACTION:")
        print("Calling retrieve_relevant_docs()")

        tool_result = retrieve_relevant_docs(question)

        print("\nOBSERVATION:")
        print(tool_result)

        messages.append({
            "role": "user",
            "content": f"Observation: {tool_result}"
        })

    # -----------------------------
    # If LLM gives final answer, stop
    # -----------------------------
    elif "Final Answer:" in reply:

        print("\nDONE.")
        break

    # -----------------------------
    # If LLM does not follow format, stop
    # -----------------------------
    else:
        print("\nLLM did not choose a valid action. Stopping.")
        break