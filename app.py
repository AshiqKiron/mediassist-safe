import streamlit as st
import json
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import FakeEmbeddings # Using fake embeddings for speed/demo since we have small data

# --- 1. SETUP & INITIALIZATION ---
st.set_page_config(page_title="MediAssist Safe", page_icon="🏥", layout="centered")

# Initialize Groq LLM
llm = ChatGroq(temperature=0.0, model_name="llama3-70b-8192", groq_api_key=st.secrets["GROQ_API"])

# --- 2. RAG DATABASE SETUP (Using FAISS) ---
@st.cache_resource
def load_knowledge_base():
    with open("data/medical_knowledge.json", "r") as f:
        data = json.load(f)
    
    docs = [Document(page_content=item["content"], metadata={"topic": item["topic"]}) for item in data]
    
    # For a demo with a small dataset, we can use a simple keyword-based retrieval or 
    # a lightweight embedding. To avoid heavy embedding model downloads on Streamlit Cloud,
    # we'll use a simple trick: FAISS with a dummy embedding for structure, 
    # but in a real app, you'd use HuggingFaceEmbeddings.
    # NOTE: For this specific demo to be robust on free cloud, let's use a simple similarity search
    # by converting text to a basic vector representation or just using the built-in FAISS from_texts.
    
    try:
        # Attempt to use a real lightweight embedding if available, otherwise fallback
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)
    except Exception:
        # Fallback for environments where downloading models is restricted/slow
        # We will use a simple "keyword" match logic for the demo if FAISS fails to init with embeddings
        st.warning("Using simplified search mode due to environment constraints.")
        return None 

    return vectorstore

vectorstore = load_knowledge_base()

# --- 3. AGENTIC WORKFLOW FUNCTIONS ---

def router_agent(query: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a medical triage router. Classify the user's query into exactly one of these categories: 'EMERGENCY', 'MEDIUM', 'LOW'.\nEMERGENCY: chest pain, difficulty breathing, severe bleeding, suicidal thoughts, loss of consciousness.\nMEDIUM: specific symptoms needing general info.\nLOW: general wellness or first aid.\nOutput ONLY the category word."),
        ("human", "{query}")
    ])
    chain = prompt | llm
    result = chain.invoke({"query": query}).content.strip().upper()
    return result

def safety_verifier(query: str, rag_context: str, proposed_answer: str) -> bool:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a strict Medical Safety Verifier. Evaluate the proposed answer to the user's query.
        BLOCK (return 'UNSAFE') if the answer contains:
        1. Specific drug dosages (e.g., 'take 500mg', '2 tablets').
        2. Definitive diagnoses (e.g., 'You have the flu', 'This is a migraine').
        3. Advice that discourages seeing a doctor for serious symptoms.
        If the answer is general, safe, first-aid information and recommends consulting a professional, return 'SAFE'."""),
        ("human", "User Query: {query}\nRetrieved Context: {context}\nProposed Answer: {answer}")
    ])
    chain = prompt | llm
    result = chain.invoke({"query": query, "context": rag_context, "answer": proposed_answer}).content.strip().upper()
    return "SAFE" in result

def generate_safe_response(query: str, context: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are MediAssist Safe, a helpful but cautious medical information assistant. 
        Use ONLY the provided context to answer. 
        NEVER provide specific dosages or definitive diagnoses. 
        ALWAYS end your response with: '⚠️ Disclaimer: This information is for educational purposes only and does not replace professional medical advice. Please consult a healthcare provider.'"""),
        ("human", "Context: {context}\nUser Query: {query}")
    ])
    chain = prompt | llm
    return chain.invoke({"context": context, "query": query}).content

# --- 4. STREAMLIT UI ---
st.title("🏥 MediAssist Safe")
st.markdown("*AI-powered symptom checking with strict safety guardrails.*")

if "accepted" not in st.session_state:
    st.session_state.accepted = False

st.session_state.accepted = st.checkbox(
    "⚠️ I understand this is an AI demo for educational purposes only and does not replace professional medical advice.",
    value=st.session_state.accepted
)

if not st.session_state.accepted:
    st.warning("Please accept the disclaimer to use MediAssist Safe.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Describe your symptoms or ask a general health question..."):
    st.session
