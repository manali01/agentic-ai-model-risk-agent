# ============================================================
# DAY 5 RAG + AGENTIC AI PROJECT (OLLAMA VERSION)
# ============================================================
#
# WHAT THIS PROJECT DOES:
#
# You are building a mini "Model Risk Review Agent".
#
# The agent can:
#
# 1. Read a model document
# 2. Break document into chunks
# 3. Store chunks in vector database
# 4. Retrieve relevant chunks when needed
# 5. Retrieve historical issues
# 6. Use standard model risk checklist
# 7. Reason step-by-step
# 8. Produce final model review
#
#
# OVERALL FLOW:
#
# USER QUESTION
#       ↓
# LLM THINKS
#       ↓
# LLM CHOOSES TOOL
#       ↓
# PYTHON EXECUTES TOOL
#       ↓
# TOOL RESULT SENT BACK
#       ↓
# LLM THINKS AGAIN
#       ↓
# FINAL ANSWER
#
#
# IMPORTANT CONCEPTS:
#
# TOOL = capability agent can use
#
# TOOL 1:
# Standard model risk checklist
#
# TOOL 2:
# Historical issue retrieval
#
# TOOL 3:
# Search uploaded model document
#
#
# ============================================================

import ollama
import chromadb


# ============================================================
# CONFIGURATION
# ============================================================

# This is the LLM (brain)
LLM_MODEL = "llama3"

# This model converts text into embeddings/vectors
EMBED_MODEL = "nomic-embed-text"

# This is the uploaded/current model document
MODEL_DOC_PATH = "large_model_document.txt"


# ============================================================
# STEP 1: READ MODEL DOCUMENT
# ============================================================
#
# This function simply opens the text file
# and reads the full document.
#
# ============================================================

def read_document(file_path):

    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


# ============================================================
# STEP 2: CHUNKING FUNCTION
# ============================================================
#
# WHY CHUNKING?
#
# Large documents cannot be sent fully to LLM.
#
# So we:
# 1. Split document into smaller pieces
# 2. Store pieces separately
# 3. Retrieve only relevant pieces later
#
#
# EXAMPLE:
#
# Large document:
# [100 pages]
#
# Becomes:
#
# Chunk 1
# Chunk 2
# Chunk 3
# ...
#
# ============================================================

def chunk_document(text, chunk_size=300):

    chunks = []

    # Loop through document in small windows
    for i in range(0, len(text), chunk_size):

        # Extract small chunk
        chunk = text[i:i + chunk_size]

        # Save chunk
        chunks.append(chunk)

    return chunks


# ============================================================
# TOOL 1: MODEL RISK RULES TOOL
# ============================================================
#
# PURPOSE:
#
# Returns standard model risk checklist.
#
# Think of this as:
# "institutional model risk knowledge"
#
# ============================================================

def get_model_risk_rules():

    return [
        "Check for data leakage",
        "Check for out-of-time validation",
        "Check for bias testing",
        "Check for feature stability",
        "Check for monitoring thresholds",
        "Check for independent validation",
    ]


# ============================================================
# TOOL 2: HISTORICAL ISSUES TOOL
# ============================================================
#
# PURPOSE:
#
# Search past model review issues.
#
# Think:
#
# "What problems happened in similar models before?"
#
#
# IMPORTANT:
#
# This is DIFFERENT from current document retrieval.
#
# This searches:
# historical issue database
#
# ============================================================

historical_issues = [

    "Model issue: out-of-time validation was missing.",

    "Model issue: feature leakage due to future payment behavior.",

    "Model issue: monitoring thresholds were not documented.",

    "Model issue: bias testing across customer groups was incomplete.",

    "Model issue: challenger model framework was weak."
]


# ============================================================
# CREATE VECTOR DATABASE
# ============================================================
#
# Chroma stores embeddings/vectors.
#
# Embeddings = numerical representation of meaning.
#
# Similar meaning → vectors close together.
#
# ============================================================

client = chromadb.PersistentClient(path="./rag_agent_db")


# ============================================================
# CREATE COLLECTION FOR HISTORICAL ISSUES
# ============================================================

issues_collection = client.get_or_create_collection(
    name="historical_issues"
)


# ============================================================
# STORE HISTORICAL ISSUES IN VECTOR DB
# ============================================================
#
# FOR EACH ISSUE:
#
# 1. Convert text → embedding
# 2. Store embedding in Chroma
#
# ============================================================

for i, issue in enumerate(historical_issues):

    # Convert text into embedding/vector
    embedding = ollama.embeddings(
        model=EMBED_MODEL,
        prompt=issue
    )["embedding"]

    # Store in vector DB
    issues_collection.upsert(
        ids=[f"issue_{i}"],
        documents=[issue],
        embeddings=[embedding]
    )


# ============================================================
# TOOL 2 FUNCTION
# ============================================================
#
# PURPOSE:
#
# Given a query:
#
# 1. Convert query → embedding
# 2. Search similar embeddings
# 3. Return most relevant historical issues
#
# ============================================================

def retrieve_similar_issues(query, top_k=3):

    # Convert query into embedding
    query_embedding = ollama.embeddings(
        model=EMBED_MODEL,
        prompt=query
    )["embedding"]

    # Search vector DB
    results = issues_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    # Return matching documents
    return results["documents"][0]


# ============================================================
# TOOL 3: CURRENT DOCUMENT RETRIEVAL TOOL
# ============================================================
#
# PURPOSE:
#
# Search uploaded/current model document.
#
# DIFFERENT FROM TOOL 2:
#
# TOOL 2:
# searches historical issue database
#
# TOOL 3:
# searches CURRENT uploaded document
#
# ============================================================

document_collection = client.get_or_create_collection(
    name="document_chunks"
)


# ============================================================
# READ FULL MODEL DOCUMENT
# ============================================================

document_text = read_document(MODEL_DOC_PATH)


# ============================================================
# SPLIT DOCUMENT INTO CHUNKS
# ============================================================

chunks = chunk_document(document_text)


# ============================================================
# STORE CHUNKS IN VECTOR DB
# ============================================================
#
# FOR EACH CHUNK:
#
# 1. Convert chunk → embedding
# 2. Store embedding
#
# ============================================================

for i, chunk in enumerate(chunks):

    embedding = ollama.embeddings(
        model=EMBED_MODEL,
        prompt=chunk
    )["embedding"]

    document_collection.upsert(
        ids=[f"chunk_{i}"],
        documents=[chunk],
        embeddings=[embedding]
    )


# ============================================================
# TOOL 3 FUNCTION
# ============================================================
#
# PURPOSE:
#
# Search uploaded model document.
#
# FLOW:
#
# Query
#   ↓
# Query embedding
#   ↓
# Search vector DB
#   ↓
# Return most relevant chunks
#
# ============================================================

def retrieve_relevant_doc_chunks(query, top_k=3):

    query_embedding = ollama.embeddings(
        model=EMBED_MODEL,
        prompt=query
    )["embedding"]

    results = document_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results["documents"][0]


# ============================================================
# AGENT MEMORY
# ============================================================
#
# messages = conversation history
#
# LLM sees this EVERY TIME.
#
# This acts as agent memory/state.
#
# ============================================================

messages = [

    # ========================================================
    # SYSTEM PROMPT
    #
    # This defines:
    # - agent role
    # - available tools
    # - expected format
    #
    # ========================================================

    {
        "role": "system",
        "content": """
You are a senior model risk reviewer.

You have 3 tools:

1. get_model_risk_rules
Use for standard checklist.

2. retrieve_similar_issues
Use for historical issue patterns.

3. retrieve_relevant_doc_chunks
Use for searching uploaded model document.

Use this format:

Thought:
Action:

Allowed actions:
Action: get_model_risk_rules
Action: retrieve_similar_issues
Action: retrieve_relevant_doc_chunks

After observations, provide:

Final Answer:

Risk Identified:
Testing Needed:
Review / Challenge:
Confidence:

Important:
- Once you provide Final Answer, do not include any Action after it.
- Do not provide multiple Final Answers.
- Keep answer concise.
"""
    },

    # ========================================================
    # USER QUESTION
    # ========================================================

    {
        "role": "user",
        "content": """
Review the uploaded model document and identify:
- risks
- testing needed
- challenge questions
"""
    }
]


# ============================================================
# AGENT LOOP
# ============================================================
#
# THIS IS THE MOST IMPORTANT PART.
#
# LOOP FLOW:
#
# 1. LLM thinks
# 2. LLM chooses tool
# 3. Python executes tool
# 4. Tool output returned
# 5. LLM thinks again
# 6. Repeat
#
#
# WHY LIMIT TO 8 STEPS?
#
# Prevent infinite loops.
#
# ============================================================

for step in range(8):

    print(f"\n========== STEP {step + 1} ==========")

    # ========================================================
    # ASK LLM TO RESPOND
    # ========================================================

    response = ollama.chat(
        model=LLM_MODEL,
        messages=messages,
    )

    # Extract LLM text
    reply = response["message"]["content"]

    print("\nLLM RESPONSE:")
    print(reply)

    # Save LLM response into memory
    messages.append({
        "role": "assistant",
        "content": reply
    })

    reply_lower = reply.lower()

     # ========================================================
    # IMPORTANT FIX:
    # Check Final Answer FIRST.
    # This prevents the agent from continuing after final answer.
    # ========================================================

    if (
        "final answer" in reply_lower
        or "risk identified:" in reply_lower
        or "testing needed:" in reply_lower
        or "review / challenge:" in reply_lower
        or "confidence:" in reply_lower
    ):
        print("\nDONE.")
        break
    # ========================================================
    # TOOL 1 SELECTED
    # ========================================================

    elif "Action: get_model_risk_rules" in reply_lower:

        print("\nCALLING TOOL: get_model_risk_rules")

        # Execute tool
        tool_result = get_model_risk_rules()

        print(tool_result)

        # Send tool result back to LLM
        messages.append({
            "role": "user",
            "content": f"Observation: {tool_result}"
        })


    # ========================================================
    # TOOL 2 SELECTED
    # ========================================================

    elif "Action: retrieve_similar_issues" in reply_lower:

        print("\nCALLING TOOL: retrieve_similar_issues")

        # Execute retrieval tool
        tool_result = retrieve_similar_issues(document_text)

        print(tool_result)

        # Return result back to LLM
        messages.append({
            "role": "user",
            "content": f"Observation: {tool_result}"
        })


    # ========================================================
    # TOOL 3 SELECTED
    # ========================================================

    elif "Action: retrieve_relevant_doc_chunks" in reply_lower:

        print("\nCALLING TOOL: retrieve_relevant_doc_chunks")

        # Execute retrieval on current document
        tool_result = retrieve_relevant_doc_chunks(
            "What are the major risks and limitations?"
        )

        print(tool_result)

        # Return result to LLM
        messages.append({
            "role": "user",
            "content": f"Observation: {tool_result}"
        })



    


    # ========================================================
    # INVALID FORMAT
    # ========================================================

    else:

        print("\nLLM did not follow expected format.")
        break