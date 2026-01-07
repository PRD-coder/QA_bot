
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def get_model(model_name="models/gemini-2.5-flash", temperature=0.3):
    """Create a Google Generative AI LLM wrapper."""
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        convert_system_message_to_human=True
    )

def choose_model(question: str, preferred_model: str):
    """
    Auto-upgrade from Flash -> Pro for complex queries.
    Keep this at top-level so imports don't fail.
    """
    q = (question or "").lower()
    # If user explicitly prefers pro, honor it
    if preferred_model == "models/gemini-2.5-pro":
        return "models/gemini-2.5-pro"
    # Heuristic triggers for pro
    if any(word in q for word in ["flowchart", "diagram", "detailed", "steps", "complex", "explain in detail"]):
        return "models/gemini-2.5-pro"
    # otherwise default to flash
    return "models/gemini-2.5-flash"

# ---------------- Chain Builders ---------------- #
def build_single_answer_chain(model_name="models/gemini-2.5-flash"):
    llm = get_model(model_name)
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a helpful assistant. Use the context below to answer the question.\n"
            "For every claim, cite the source using the metadata key 'source' from the context, formatted as [source].\n\n"
            "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )
    )
    return prompt | llm | StrOutputParser()

def build_steps_answer_chain(model_name="models/gemini-2.5-flash"):
    llm = get_model(model_name)
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a step-by-step instructor. Use the context to provide clear steps.\n"
            "For every claim, cite the source using the metadata key 'source', formatted as [source].\n\n"
            "Context:\n{context}\n\nQuestion: {question}\n\nSteps:"
        )
    )
    return prompt | llm | StrOutputParser()

def build_multi_answer_chain(model_name="models/gemini-2.5-flash"):
    llm = get_model(model_name)
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Provide multiple styles of answers with sources for each claim.\n\n"
            "Context:\n{context}\n\nQuestion: {question}\n\n"
            "**Technical Answer** (with citations):\n.\n\n"
            "**Summary Answer** (with citations):\n.\n\n"
            "**Simple Answer** (with citations):\n."
        )
    )
    return prompt | llm | StrOutputParser()

def build_flowchart_answer_chain(model_name="models/gemini-2.5-pro"):
    llm = get_model(model_name)
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a system that outputs flowcharts in Mermaid.js syntax only.\n"
            "Base your answer strictly on the context, and cite the sources in comments.\n\n"
            "Context:\n{context}\n\nQuestion: {question}\n\nFlowchart:"
        )
    )
    return prompt | llm | StrOutputParser()

def build_general_chat_chain(model_name="models/gemini-2.5-flash"):
    llm = get_model(model_name)
    prompt = PromptTemplate(
        input_variables=["question"],
        template=(
            "You are ChatGPT-like assistant. Answer the question comprehensively.\n\n"
            "Question: {question}\n\nAnswer:"
        )
    )
    return prompt | llm | StrOutputParser()

# ---------------- Retrievers ---------------- #
def get_metadata_retriever(vectorstore, k=4):
    """Fast retriever based on metadata."""
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})

def get_advanced_retriever(vectorstore, k=4):
    """Deep retriever with max marginal relevance."""
    return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": k})
