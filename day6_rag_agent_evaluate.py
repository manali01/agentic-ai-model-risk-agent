# ============================================================
# DAY 6.5 — FULL AGENT + EVALUATION
# FIXED VERSION WITH FLEXIBLE ACTION DETECTION
# ============================================================

import ollama
import chromadb


# ============================================================
# CONFIG
# ============================================================

LLM_MODEL = "llama3"
EMBED_MODEL = "nomic-embed-text"


# ============================================================
# TEST FILES
# ============================================================

TEST_CASES = [
    {
        "file": "test_docs/good_model.txt",
        "expected_risks": [
            "no major risk"
        ]
    },
    {
        "file": "test_docs/weak_model.txt",
        "expected_risks": [
            "data leakage",
            "missing out-of-time validation",
            "missing bias testing",
            "missing monitoring thresholds",
            "missing feature stability",
            "weak independent validation"
        ]
    },
    {
        "file": "test_docs/average_model.txt",
        "expected_risks": [
            "incomplete out-of-time validation",
            "limited bias testing",
            "partial monitoring thresholds",
            "limited feature stability"
        ]
    }
]


# ============================================================
# READ FILE
# ============================================================

def read_file(path):
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


# ============================================================
# CHUNK DOCUMENT
# ============================================================

def chunk_document(text, chunk_size=300):
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])

    return chunks


# ============================================================
# TOOL 1 — MODEL RISK RULES
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
# TOOL 2 — HISTORICAL ISSUES
# ============================================================

HISTORICAL_ISSUES = [
    "Historical issue: model used recent payment behavior that caused possible data leakage.",
    "Historical issue: out-of-time validation was missing or poorly documented.",
    "Historical issue: bias testing across customer segments was incomplete.",
    "Historical issue: monitoring thresholds were not documented.",
    "Historical issue: feature stability analysis was missing.",
    "Historical issue: independent validation signoff was incomplete.",
]


# ============================================================
# BUILD VECTOR COLLECTION
# ============================================================

def build_collection(collection_name, documents):
    client = chromadb.PersistentClient(path="./day6_5_chroma_db")
    collection = client.get_or_create_collection(name=collection_name)

    for i, doc in enumerate(documents):
        embedding = ollama.embeddings(
            model=EMBED_MODEL,
            prompt=doc
        )["embedding"]

        collection.upsert(
            ids=[f"{collection_name}_{i}"],
            documents=[doc],
            embeddings=[embedding]
        )

    return collection


# ============================================================
# RETRIEVE FROM VECTOR DATABASE
# ============================================================

def retrieve_from_collection(collection, query, top_k=3):
    query_embedding = ollama.embeddings(
        model=EMBED_MODEL,
        prompt=query
    )["embedding"]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results["documents"][0]


# ============================================================
# FULL AGENT
# ============================================================

def run_full_agent(document_text):

    # Chunk the current model document
    doc_chunks = chunk_document(document_text)

    # Build retriever for current document chunks
    doc_collection = build_collection(
        collection_name="current_doc_chunks",
        documents=doc_chunks
    )

    # Build retriever for historical issues
    issues_collection = build_collection(
        collection_name="historical_issues",
        documents=HISTORICAL_ISSUES
    )

    # Track which tools were actually executed
    tools_used = []

    # Conversation memory
    messages = [
        {
            "role": "system",
            "content": """
You are a senior model risk reviewer.

You have 3 tools:

1. get_model_risk_rules
Use this for standard model risk checklist.

2. retrieve_relevant_doc_chunks
Use this to search CURRENT model document.

3. retrieve_similar_issues
Use this to search historical model review issues.

VERY IMPORTANT:
Never include Action and Final Answer in the same response.

When using a tool, write exactly:

Thought:
Action: tool_name

Allowed actions:

Action: get_model_risk_rules
Action: retrieve_relevant_doc_chunks
Action: retrieve_similar_issues

Do not write Final Answer until after observations are returned.

Final answer format:

Final Answer:

Risks:
- risk 1
- risk 2

Testing Needed:
- test 1
- test 2

Challenge Questions:
- question 1
- question 2

Be specific to the model document.
Avoid generic risks.
"""
        },
        {
            "role": "user",
            "content": f"""
Review this model document:

{document_text}

Identify:
- risks
- testing needed
- challenge questions
"""
        }
    ]

    # Agent loop
    for step in range(8):

        print(f"\n========== STEP {step + 1} ==========")

        response = ollama.chat(
            model=LLM_MODEL,
            messages=messages,
        )

        reply = response["message"]["content"]

        print("\nLLM RESPONSE:")
        print(reply)

        messages.append({
            "role": "assistant",
            "content": reply
        })

        # ====================================================
        # NORMALIZE LLM REPLY
        # Handles both:
        # Action: get_model_risk_rules
        # and:
        # Action:
        # get_model_risk_rules
        # ====================================================

        reply_lower = reply.lower()

        has_final_answer = "final answer:" in reply_lower
        has_action = "action:" in reply_lower

        wants_risk_rules = (
            "action: get_model_risk_rules" in reply_lower
            or ("action:" in reply_lower and "get_model_risk_rules" in reply_lower)
        )

        wants_doc_retriever = (
            "action: retrieve_relevant_doc_chunks" in reply_lower
            or ("action:" in reply_lower and "retrieve_relevant_doc_chunks" in reply_lower)
        )

        wants_historical_issues = (
            "action: retrieve_similar_issues" in reply_lower
            or ("action:" in reply_lower and "retrieve_similar_issues" in reply_lower)
        )

        # ====================================================
        # INVALID: ACTION + FINAL ANSWER TOGETHER
        # ====================================================

        if has_final_answer and has_action:

            print("\nINVALID RESPONSE")
            print("LLM included BOTH Action and Final Answer.")

            messages.append({
                "role": "user",
                "content": """
Invalid response.

You included BOTH Action and Final Answer.

Now respond with ONLY ONE of these formats.

If tool is needed:
Thought:
Action: get_model_risk_rules

OR

Thought:
Action: retrieve_relevant_doc_chunks

OR

Thought:
Action: retrieve_similar_issues

If no more tool is needed:
Final Answer:

Do not include explanation.
Do not apologize.
Do not include both Action and Final Answer.
"""
            })

            continue

        # ====================================================
        # VALID FINAL ANSWER
        # ====================================================

        elif has_final_answer:

            return {
                "agent_output": reply,
                "tools_used": tools_used,
                "messages": messages
            }

        # ====================================================
        # TOOL 1 — RISK RULES
        # ====================================================

        elif wants_risk_rules:

            if "get_model_risk_rules" in tools_used:
                messages.append({
                    "role": "user",
                    "content": """
Observation:
get_model_risk_rules already used.

Do not call it again.
Provide Final Answer.
"""
                })
                continue

            print("\nCALLING TOOL: get_model_risk_rules")

            tools_used.append("get_model_risk_rules")

            tool_result = get_model_risk_rules()

            print(tool_result)

            messages.append({
                "role": "user",
                "content": f"""
Observation from get_model_risk_rules:

{tool_result}
"""
            })

        # ====================================================
        # TOOL 2 — CURRENT DOCUMENT RETRIEVER
        # ====================================================

        elif wants_doc_retriever:

            if "retrieve_relevant_doc_chunks" in tools_used:
                messages.append({
                    "role": "user",
                    "content": """
Observation:
retrieve_relevant_doc_chunks already used.

Do not call it again.
Provide Final Answer.
"""
                })
                continue

            print("\nCALLING TOOL: retrieve_relevant_doc_chunks")

            tools_used.append("retrieve_relevant_doc_chunks")

            tool_result = retrieve_from_collection(
                collection=doc_collection,
                query="""
                risks limitations validation bias monitoring
                feature stability independent validation data leakage
                """,
                top_k=3
            )

            print(tool_result)

            messages.append({
                "role": "user",
                "content": f"""
Observation from retrieve_relevant_doc_chunks:

{tool_result}
"""
            })

        # ====================================================
        # TOOL 3 — HISTORICAL ISSUES
        # ====================================================

        elif wants_historical_issues:

            if "retrieve_similar_issues" in tools_used:
                messages.append({
                    "role": "user",
                    "content": """
Observation:
retrieve_similar_issues already used.

Do not call it again.
Provide Final Answer.
"""
                })
                continue

            print("\nCALLING TOOL: retrieve_similar_issues")

            tools_used.append("retrieve_similar_issues")

            tool_result = retrieve_from_collection(
                collection=issues_collection,
                query=document_text,
                top_k=3
            )

            print(tool_result)

            messages.append({
                "role": "user",
                "content": f"""
Observation from retrieve_similar_issues:

{tool_result}
"""
            })

        # ====================================================
        # INVALID FORMAT
        # ====================================================

        else:

            print("\nINVALID FORMAT")
            print("LLM did not follow expected structure.")

            messages.append({
                "role": "user",
                "content": """
Invalid format.

Respond using exactly one of these:

Thought:
Action: get_model_risk_rules

Thought:
Action: retrieve_relevant_doc_chunks

Thought:
Action: retrieve_similar_issues

Final Answer:
"""
            })

            continue

    # If loop ends without clean final answer
    return {
        "agent_output": messages[-1]["content"],
        "tools_used": tools_used,
        "messages": messages
    }


# ============================================================
# EXTRACT RISKS
# ============================================================

def extract_risks(agent_output):
    risks = []
    capture = False

    for line in agent_output.splitlines():
        clean_line = line.strip().lower()

        if clean_line.startswith("risks"):
            capture = True
            continue

        if clean_line.startswith("testing needed"):
            capture = False

        if capture and clean_line.startswith("-"):
            risk = clean_line.replace("-", "").strip()
            risks.append(risk)

    return risks


# ============================================================
# RISK MATCHING
# ============================================================

def risk_matches(expected_risk, predicted_risks):
    expected_words = set(expected_risk.lower().split())

    for predicted in predicted_risks:
        predicted_words = set(predicted.lower().split())
        overlap = expected_words.intersection(predicted_words)

        if len(overlap) >= 2:
            return True

    return False


# ============================================================
# EVALUATE ONE TEST CASE
# ============================================================

def evaluate_case(test_case):

    document_text = read_file(test_case["file"])
    expected_risks = test_case["expected_risks"]

    print("\n====================================")
    print(f"Evaluating: {test_case['file']}")
    print("====================================")

    agent_result = run_full_agent(document_text)

    agent_output = agent_result["agent_output"]
    tools_used = agent_result["tools_used"]

    print("\nTOOLS USED:")
    print(tools_used)

    print("\nAGENT OUTPUT:")
    print(agent_output)

    predicted_risks = extract_risks(agent_output)

    print("\nPREDICTED RISKS:")
    print(predicted_risks)

    matched = []
    missed = []

    for expected in expected_risks:
        if risk_matches(expected, predicted_risks):
            matched.append(expected)
        else:
            missed.append(expected)

    false_positives = []

    for predicted in predicted_risks:
        found_match = False

        for expected in expected_risks:
            if risk_matches(expected, [predicted]):
                found_match = True
                break

        if not found_match:
            false_positives.append(predicted)

    precision = len(matched) / max(len(predicted_risks), 1)
    recall = len(matched) / max(len(expected_risks), 1)

    print("\nMATCHED RISKS:")
    print(matched)

    print("\nMISSED RISKS:")
    print(missed)

    print("\nFALSE POSITIVES:")
    print(false_positives)

    print("\nMETRICS:")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

    return {
        "file": test_case["file"],
        "precision": precision,
        "recall": recall,
        "matched": matched,
        "missed": missed,
        "false_positives": false_positives,
        "tools_used": tools_used
    }


# ============================================================
# MAIN
# ============================================================

all_results = []

for case in TEST_CASES:
    result = evaluate_case(case)
    all_results.append(result)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n\n========== FINAL SUMMARY ==========")

avg_precision = sum(r["precision"] for r in all_results) / len(all_results)
avg_recall = sum(r["recall"] for r in all_results) / len(all_results)

print(f"Average Precision: {avg_precision:.2f}")
print(f"Average Recall: {avg_recall:.2f}")

print("\nTOOL USAGE SUMMARY:")
for result in all_results:
    print(f"{result['file']}: {result['tools_used']}")