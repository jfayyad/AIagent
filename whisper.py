import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_google_genai import ChatGoogleGenerativeAI

# Load secrets from environment (Streamlit will inject GOOGLE_API_KEY)
load_dotenv()
st.set_page_config(page_title="Gemini Whisper Bot", layout="wide")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

st.title("🧠 Gemini Bot with Whispered Guidance")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

# Layout: User input left | Advisor whisper right
col1, col2 = st.columns(2)

with col1:
    st.subheader("🧑 User")
    user_input = st.text_input("Type your message:", key="user_input")
    send_button = st.button("Send")

with col2:
    st.subheader("👤 Advisor (Supervisor)")
    whisper = st.text_input("Whisper to the bot:", key="whisper_input")

# Conversation prompt template
prompt = PromptTemplate.from_template("""
You are a helpful, warm customer support assistant.

You MUST follow any supervisor advice given below exactly and prioritize it in your reply.

Context:
- Past memory: User has had password trouble on mobile before.
- Supervisor advice (optional): {whisper}

Recent conversation:
{history}

User: {user_input}
Assistant:
""")

# If user sends message
if send_button and user_input.strip():
    pipeline = (
        RunnableMap({
            "whisper": lambda _: whisper,
            "user_input": lambda _: user_input,
            "history": lambda _: "\n".join(st.session_state.get("history", [])[-6:]),
        })
        | prompt
        | llm
    )

    response = pipeline.invoke({})
    st.session_state.history.append(f"user: {user_input}")
    st.session_state.history.append(f"assistant: {response.content}")

# Display full conversation history
for msg in st.session_state.history:
    role, content = msg.split(": ", 1)
    with st.chat_message(role):
        st.markdown(content)
