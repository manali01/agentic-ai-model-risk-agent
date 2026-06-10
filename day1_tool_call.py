import ollama

def get_model_risk_rules():
    return [
        "Data leakage",
        "Overfitting",
        "Bias in training data",
        "Weak independent validation",
        "Feature instability",
    ]

rules = get_model_risk_rules()

prompt = f"""
Use these rules: {rules}

For a credit risk model, return only 3 bullets:
Risk | Why it matters | Testing needed
"""

stream = ollama.chat(
    model="llama3",
    messages=[{"role": "user", "content": prompt}],
    stream=True,
)

for chunk in stream:
    print(chunk["message"]["content"], end="", flush=True)