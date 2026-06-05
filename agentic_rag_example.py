from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END


model_path = "./Qwen2.5-72B-Instruct-Q4_K_M.gguf"

llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=-1,
    n_ctx=4096,
    temperature=0,
    verbose=False
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma.from_texts(
    ["Agentic RAG systems are controlled with langgraph.", "GGUF models work with llama-cpp."],
    embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

class GraphState(TypedDict):
    question: str
    generation: str
    context: str
    route: str

def router(state: GraphState):
    question = state["question"].lower()
    if any(word in question for word in ["what", "how", "information"]):
        return {"route": "retrieve"}
    return {"route": "direct_answer"}

def retrieve(state: GraphState):
    docs = retriever.invoke(state["question"])
    return {"context": "\n".join([d.page_content for d in docs])}

def generate(state: GraphState):
    context = state.get("context", "Genel bilgi.")
    prompt = f"Context: {context}\nQuestion: {state['question']}\nAnswer:"
    res = llm.invoke(prompt)
    return {"generation": res}

workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

workflow.set_node_alpha_border = "router"

def route_decision(state: GraphState):
    if state["route"] == "retrieve":
        return "retrieve"
    return "generate"

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

result = app.invoke({"question": "Agentic RAG nedir?"})
print(f"Sonuç: {result['generation']}")