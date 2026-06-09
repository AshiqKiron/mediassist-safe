import streamlit as st
import json
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# --- 1. SETUP & INITIALIZATION ---
st.set_page_config(page_title="MediAssist Safe", page_icon="🏥", layout="centered")

# Initialize Groq LLM with the latest stable model
try:
    llm = ChatGroq(temperature=0.0, model_name="llama-3.3-70b-versatile", groq_api_key=st.secrets["GROQ_API"])
except Exception as e:
    st.error(f"Failed to initialize LLM: {e}")
    st.stop()

# --- 2. LOAD DATA ---
@st.cache_data
def load_knowledge_data():
    with open("data/medical_knowledge.json", "r") as f:
        return json.load(f)

medical_data = load_knowledge_data()

# --- 3. AGENTIC WORKFLOW FUNCTIONS ---

def router_agent(query: str) -> str:
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a medical triage router. Classify the user's query into exactly one of these categories: 'EMERGENCY', 'MEDIUM', 'LOW'.\nEMERGENCY: chest pain, difficulty breathing, severe bleeding, suicidal thoughts, loss of consciousness.\nMEDIUM: specific symptoms needing general info.\nLOW: general wellness or first aid.\nOutput ONLY the category word."),
            ("human", "{query}")
        ])
        chain = prompt | llm
        result = chain.invoke({"query": query}).content.strip().upper()
        # Ensure we only get one of the expected words
        if "EMERGENCY" in result: return "EMERGENCY"
        if "MEDIUM" in result: return "MEDIUM"
        return "LOW"
    except Exception as e:
        st.warning(f"Router Agent Error: {e}")
        return "MEDIUM" # Default to medium for safety

def safety_verifier(query: str, rag_context: str, proposed_answer: str) -> bool:
    try:
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
    except Exception as e:
        st.warning(f"Safety Verifier Error: {e}")
        return False # Fail safe: block if verifier crashes

def generate_safe_response(query: str, context: str) -> str:
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are MediAssist Safe, a helpful but cautious medical information assistant. 
            Use ONLY the provided context to answer. 
            NEVER provide specific dosages or definitive diagnoses. 
            ALWAYS end your response with: '⚠️ Disclaimer: This information is for educational purposes only and does not replace professional medical advice. Please consult a healthcare provider.'"""),
            ("human", "Context: {context}\nUser Query: {query}")
        ])
        chain = prompt | llm
        return chain.invoke({"context": context, "query": query}).content
    except Exception as e:
        return f"I encountered an error processing your request: {e}. Please try again."

def simple_keyword_search(query: str, data: list, k: int = 2) -> str:
    """A simple fallback search if vector stores fail."""
    query_words = set(query.lower().split())
    scored_items = []
    for item in data:
        topic_words = set(item["topic"].lower().split())
        content_words = set(item["content"].lower().split())
        score = len(query_words.intersection(topic_words)) * 2 + len(query_words.intersection(content_words))
        if score > 0:
            scored_items.append((score, item["content"]))
    
    scored_items.sort(key=lambda x: x[0], reverse=True)
    results = [item[1] for item in scored_items[:k]]
    if not results:
        return "No specific information found in the database. Please consult a healthcare provider."
    return "\n\n".join(results)

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
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Triage and Safety Verification in progress..."):
            urgency = router_agent(prompt)
            
            if urgency == "EMERGENCY":
                response = "🚨 **EMERGENCY DETECTED** 🚨\n\nBased on your description, this may be a medical emergency. Please stop using this app and **call emergency services (e.g., 911) or go to the nearest emergency room immediately.**"
            else:
                context = simple_keyword_search(prompt, medical_data, k=2)
                draft_answer = generate_safe_response(prompt, context)
                is_safe = safety_verifier(prompt, context, draft_answer)
                
                if not is_safe:
                    response = "🛑 **Safety Guardrail Triggered**\n\nI cannot provide specific dosages, definitive diagnoses, or individualized treatment plans. Please consult a licensed pharmacist or healthcare provider for personalized medical advice."
                else:
                    response = draft_answer

            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
