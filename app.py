import streamlit as st
import json
import os
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# --- 1. SETUP & INITIALIZATION ---
st.set_page_config(page_title="MediAssist Safe", page_icon="🏥", layout="centered")

# Initialize Groq LLM (using llama3-70b-8192 for high reasoning capability)
llm = ChatGroq(temperature=0.0, model_name="llama3-70b-8192", groq_api_key=st.secrets["GROQ_API"])

# --- 2. RAG DATABASE SETUP ---
@st.cache_resource
def load_knowledge_base():
    """Loads the dummy JSON data into a local ChromaDB vector store."""
    with open("data/medical_knowledge.json", "r") as f:
        data = json.load(f)
    
    docs = [Document(page_content=item["content"], metadata={"topic": item["topic"]}) for item in data]
    
    # Use a local persistent directory. Streamlit Cloud will rebuild this on each cold start, which is fine for a demo.
    vectorstore = Chroma.from_documents(documents=docs, collection_name="mediassist_db", persist_directory="./chroma_db")
    return vectorstore

vectorstore = load_knowledge_base()
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# --- 3. AGENTIC WORKFLOW FUNCTIONS ---

def router_agent(query: str) -> str:
    """Classifies urgency. Blocks EMERGENCY immediately."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a medical triage router. Classify the user's query into exactly one of these categories: 'EMERGENCY', 'MEDIUM', 'LOW'.\nEMERGENCY: chest pain, difficulty breathing, severe bleeding, suicidal thoughts, loss of consciousness.\nMEDIUM: specific symptoms needing general info.\nLOW: general wellness or first aid.\nOutput ONLY the category word."),
        ("human", "{query}")
    ])
    chain = prompt | llm
    result = chain.invoke({"query": query}).content.strip().upper()
    return result

def safety_verifier(query: str, rag_context: str, proposed_answer: str) -> bool:
    """LLM-as-a-Judge to block dangerous advice."""
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
    """Generates the final response, enforcing disclaimers."""
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

# Mandatory Disclaimer Checkbox
if "accepted" not in st.session_state:
    st.session_state.accepted = False

st.session_state.accepted = st.checkbox(
    "⚠️ I understand this is an AI demo for educational purposes only and does not replace professional medical advice.",
    value=st.session_state.accepted
)

if not st.session_state.accepted:
    st.warning("Please accept the disclaimer to use MediAssist Safe.")
    st.stop()

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Describe your symptoms or ask a general health question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Triage and Safety Verification in progress..."):
            
            # STEP 1: Router Agent
            urgency = router_agent(prompt)
            
            if urgency == "EMERGENCY":
                response = "🚨 **EMERGENCY DETECTED** 🚨\n\nBased on your description, this may be a medical emergency. Please stop using this app and **call emergency services (e.g., 911) or go to the nearest emergency room immediately.**"
            else:
                # STEP 2: RAG Agent
                docs = retriever.invoke(prompt)
                context = "\n\n".join([d.page_content for d in docs])
                
                # STEP 3: Draft Answer
                draft_answer = generate_safe_response(prompt, context)
                
                # STEP 4: Safety Verifier Agent
                is_safe = safety_verifier(prompt, context, draft_answer)
                
                if not is_safe:
                    response = "🛑 **Safety Guardrail Triggered**\n\nI cannot provide specific dosages, definitive diagnoses, or individualized treatment plans. Please consult a licensed pharmacist or healthcare provider for personalized medical advice."
                else:
                    response = draft_answer

            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})