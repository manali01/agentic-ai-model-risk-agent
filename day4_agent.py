import ollama
import chromadb


# ==================================================
# TOOL 1: Model Risk Rules Tool
# This is a simple checklist tool.
# ==================================================
def get_model_risk_rules():
    return [
        "Data leakage",
        "Overfitting",
        "Bias in training data",
        "Weak independent validation",
        "Feature instability",
    ]


# ==================================================
# TOOL 2: Retriever Tool
# This searches your small model-risk knowledge base.
# ==================================================

docs = [
    "Credit risk model using logistic regression with income, credit score, utilization, and delinquency history.",
    "Marketing model using XGBoost for customer response prediction.",
    "Validation includes out-of-time testing, backtesting, stability testing, and challenger model comparison.",
    "Risk: feature leakage if future payment behavior is included in training data.",
    "Risk: bias if model performance is not tested across customer segments.",
]

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="model_docs")


# Add documents to Chroma vector database
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


# ==================================================
# USER QUESTION
# ==================================================
question = "What risks should I check for a credit risk model?"


# ==================================================
# AGENT MEMORY
# This stores conversation history.
# ==================================================
messages = [
    {
        "role": "system",
        "content": """
You are a model risk reviewer.

You have two tools:

1. get_model_risk_rules
Use this for standard model risk checklist.

2. retrieve_relevant_docs
Use this for relevant past model documentation or issues.

Use this exact format:

Thought: explain what you need
Action: tool_name

Allowed actions:
Action: get_model_risk_rules
Action: retrieve_relevant_docs

After receiving an Observation, either use another tool or provide:

Final Answer:
Risk:
Why it matters:
Testing needed:

Do not invent tool names.
Keep the final answer short.
"""
    },
    {
        "role": "user",
        "content": question
    }
]


# ==================================================
# AGENT LOOP
# LLM decides tool.
# Python executes tool.
# Tool result goes back to LLM.
# ==================================================
for step in range(5):

    print(f"\n========== STEP {step + 1} ==========")

    response = ollama.chat(
        model="llama3",
        messages=messages,
    )

    reply = response["message"]["content"]

    print("\nLLM RESPONSE:")
    print(reply)

    messages.append({
        "role": "assistant",
        "content": reply
    })

    # ------------------------------
    # Tool 1 selected
    # ------------------------------
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

    # ------------------------------
    # Tool 2 selected
    # ------------------------------
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

    # ------------------------------
    # Final answer
    # ------------------------------
    elif "Final Answer:" in reply:

        print("\nDONE.")
        break

    else:
        print("\nLLM did not follow expected format. Stopping.")
        break