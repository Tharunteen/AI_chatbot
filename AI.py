import os
import streamlit as st
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# --- Streamlit Page Config ---
st.set_page_config(page_title="NVIDIA AI Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 NVIDIA AI Chatbot")
st.markdown("Chat with NVIDIA-hosted LLMs 🚀")

# --- Sidebar Settings ---
st.sidebar.header("⚙️ Model Settings")

# Only include NVIDIA NIM models here
model = st.sidebar.selectbox(
    "Choose Model",
    [
        "nvidia/nvidia-nemotron-nano-9b-v2",
        "nvidia/nemotron-4-340b-instruct",
        "nvidia/llama-3-70b-instruct"
    ],
    index=0,
)

temperature = st.sidebar.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
top_p = st.sidebar.slider("Top-p (nucleus sampling)", 0.1, 1.0, 0.9, 0.05)
top_k = st.sidebar.slider("Top-k (sample from k tokens)", 1, 100, 50, 1)
repetition_penalty = st.sidebar.slider("Repetition Penalty", 0.5, 2.0, 1.0, 0.05)
max_output_tokens = st.sidebar.slider("Max Output Tokens", 64, 4096, 512, 64)

# Reset chat button
if st.sidebar.button("🗑️ Reset Chat"):
    st.session_state.messages = []

# Load API key from Streamlit secrets (you must set it in Streamlit Cloud > Settings > Secrets)
api_key = st.secrets.get("NVIDIA_API_KEY")

# Initialize LLM with chosen settings
llm = ChatNVIDIA(
    model=model,
    temperature=temperature,
    top_p=top_p,
    top_k=top_k,
    repetition_penalty=repetition_penalty,
    max_output_tokens=max_output_tokens,
    api_key=api_key,
)

# --- Chat Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

# --- Chat Input ---
if prompt := st.chat_input("Type your message..."):
    # Store user message
    st.session_state.messages.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = llm.invoke(prompt)
            st.markdown(response.content)

    # Store assistant response
    st.session_state.messages.append(("assistant", response.content))
