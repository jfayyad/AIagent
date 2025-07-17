import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
st.set_page_config(page_title="Gemini Whisper Test", layout="wide")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

st.title("🧠 Gemini Bot with Whispered Guidance")

# Layout: User input left | Advisor whisper right
col1, col2 = st.columns(2)

with col1:
    st.subheader("🧑 User")
    user_input = st.text_input("Type your message:", key="user_input")
    send_button = st.button("Send")

with col2:
    st.subheader("👤 Advisor (Supervisor)")
    whisper = st.text_input("Whisper to the bot:", key="whisper_input")

if "history" not in st.session_state:
    st.session_state.history = []

if send_button and user_input.strip():
    # Define conversation template
    prompt = PromptTemplate.from_template("""
You are a helpful, warm customer support agent.

Context:
- Past memory: User has had password trouble on mobile before.
- Supervisor advice (optional): {whisper}

Recent conversation:
{history}

User: {user_input}
Assistant:
""")

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
    st.session_state.history.append(f"User: {user_input}")
    st.session_state.history.append(f"Bot: {response.content}")

    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        st.markdown(response.content)
