import ollama

# ✅ Tool #1
def get_model_risk_rules():
    return [
        "Data leakage",
        "Overfitting",
        "Bias in training data",
        "Weak independent validation",
        "Feature instability",
    ]

# ✅ Call the tool manually
rules = get_model_risk_rules()

# ✅ Prompt (inject tool output)
prompt = f"""
You are a model risk reviewer.

Question:
What risks should I check for a credit risk model?

Use these rules:
{rules}

Return:
Risk | Why it matters | Testing needed
Keep it short (3 points max).
"""

# ✅ Streaming response
stream = ollama.chat(
    model="llama3",   # make sure this exists via `ollama list`
    messages=[{"role": "user", "content": prompt}],
    stream=True,
)

# ✅ Print live output
for chunk in stream:
    print(chunk["message"]["content"], end="", flush=True)