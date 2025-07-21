import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_google_genai import ChatGoogleGenerativeAI
from textblob import TextBlob  # NEW

load_dotenv()
st.set_page_config(page_title="Gemini Whisper Bot", layout="wide")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

st.title("Gemini Bot with Whispered Guidance")

if "history" not in st.session_state:
    st.session_state.history = []
if "sentiments" not in st.session_state:
    st.session_state.sentiments = []

col1, col2 = st.columns(2)

with col1:
    st.subheader("User")
    user_input = st.text_input("Type your message:", key="user_input")
    send_button = st.button("Send")

with col2:
    st.subheader("Advisor")
    whisper = st.text_input("Whisper to the bot:", key="whisper_input")

def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.2:
        return "Positive"
    elif polarity < -0.2:
        return "Negative"
    else:
        return "Neutral"

prompt = PromptTemplate.from_template("""
You are a helpful, warm customer support assistant.

You MUST follow any supervisor advice given below exactly and prioritize it in your reply.

Context:
- Past memory: User has had password trouble on mobile before.
- Supervisor advice (optional): {whisper}
- User sentiment: {sentiment}

Recent conversation:
{history}

User: {user_input}
Assistant:
""")

if send_button and user_input.strip():
    sentiment = analyze_sentiment(user_input)
    st.session_state.sentiments.append(sentiment)

    pipeline = (
        RunnableMap({
            "whisper": lambda _: whisper,
            "user_input": lambda _: user_input,
            "history": lambda _: "\n".join(st.session_state.get("history", [])[-6:]),
            "sentiment": lambda _: sentiment,
        })
        | prompt
        | llm
    )

    response = pipeline.invoke({})
    st.session_state.history.append(f"user: {user_input}")
    st.session_state.history.append(f"assistant: {response.content}")

for i, msg in enumerate(st.session_state.history):
    role, content = msg.split(": ", 1)
    with st.chat_message(role):
        st.markdown(content)
        # Display sentiment after user messages
        if role == "user":
            sentiment = st.session_state.sentiments[i//2]  # every user msg is at even idx
            st.caption(f"Sentiment: {sentiment}")
