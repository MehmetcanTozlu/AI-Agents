# AI Agents & LLM Workflows

This repository contains practical examples and experiments demonstrating how to build intelligent AI agents, Retrieval-Augmented Generation (RAG) pipelines, and structured LLM chains. The projects heavily utilize **LangChain**, **LangGraph**, and Hugging Face's **Smolagents**.

## 🚀 Key Implementations

### 1. Agentic RAG via LangGraph
A state-driven RAG architecture that routes queries to determine the best execution path.
* **Orchestration:** Built with `langgraph` (`StateGraph`).
* **Local Inference:** Utilizes `LlamaCpp` for running local GGUF models (e.g., *Qwen2.5-72B*).
* **Retrieval System:** Combines Hugging Face embeddings (`sentence-transformers/all-MiniLM-L6-v2`) with a local `Chroma` vector store.
* **Logic:** Dynamically routes questions between a retriever node and a direct generation node based on the context of the user's prompt.

### 2. Smolagents: Vision & Tool Calling
Explorations of Hugging Face's lightweight `smolagents` ecosystem using `CodeAgent`.
* **Multi-modal Capabilities:** Fetches images from the web and uses `InferenceClientModel` to analyze visual components (e.g., analyzing costume design and identifying comic characters like The Joker).
* **Custom Tool Integration:** Demonstrates how to create and inject custom Python functions (e.g., `@tool` for finding catering services) into an agent powered by a `TransformersModel` running on CUDA.

### 3. LangChain Fundamentals
Core implementations of LLM pipelines using LangChain Expression Language (LCEL).
* **Remote Inference:** Connects to remote models (e.g., *Mistral-7B-Instruct*) via `HuggingFaceHub`.
* **Prompting:** Utilizes `ChatPromptTemplate` for system and human message structuring.
* **Execution:** Showcases both single `invoke` and simulated `stream` outputs for real-time response generation.

## 🛠️ Prerequisites & Installation

To run the scripts, ensure you have Python installed along with the necessary dependencies:

```bash
pip install langchain langchain-community langgraph chromadb sentence-transformers llama-cpp-python smolagents transformers pillow requests
