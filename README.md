## Overview

This project explores how Agentic AI can be applied to model risk review workflows.

The prototype combines:

- Local LLM inference using Ollama
- Retrieval-Augmented Generation (RAG)
- Tool calling
- ReAct-style reasoning loops
- Vector search using ChromaDB
- Simple agent evaluation metrics

The objective was to understand how modern AI agents reason, retrieve information, use tools, and generate structured review outputs in a model risk management context.

## Architecture

User Request
        ↓
Model Document
        ↓
Chunking
        ↓
Embeddings
        ↓
Vector Database (ChromaDB)
        ↓
Retriever Tool
        ↓
Agent Loop
        ↓
Model Risk Rules Tool
        ↓
Historical Issues Tool
        ↓
Ollama (Llama 3)
        ↓
Structured Review Output


## Features

### ReAct Agent

Implements a Thought → Action → Observation → Final Answer workflow.

### Retrieval-Augmented Generation (RAG)

Indexes and retrieves relevant document chunks using vector embeddings.

### Tool Calling

Agent dynamically selects among:

- Model Risk Rules Tool
- Document Retriever Tool
- Historical Issues Retriever Tool

### Evaluation Framework

Evaluates:

- Precision
- Recall
- Missed risks
- False positives
- Tool utilization

## Technologies

- Python
- Ollama
- Llama 3
- ChromaDB
- Vector Embeddings
- Retrieval-Augmented Generation (RAG)

- ## Example Review Output

Risks:
- Missing out-of-time validation
- Incomplete bias testing
- Weak monitoring thresholds

Testing Needed:
- Temporal validation
- Segment-level fairness testing
- Stability monitoring

Challenge Questions:
- How was feature stability assessed?
- What monitoring triggers require model review?


## Lessons Learned

Building an agent is relatively straightforward.

Building a reliable agent is significantly harder.

Key challenges observed:

- Tool selection errors
- Hallucinated findings
- Retrieval quality issues
- Prompt sensitivity
- Evaluation complexity

## Future Enhancements

- LangGraph workflow orchestration
- Human-in-the-loop review
- Automated issue generation
- Review report generation
- Agent observability and monitoring
- Stronger evaluation framework
