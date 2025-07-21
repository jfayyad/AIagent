import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_google_genai import ChatGoogleGenerativeAI
from transformers import pipeline

load_dotenv()
st.set_page_config(page_title="Gemini Whisper Bot", layout="wide")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

st.title("Gemini Bot with Whispered Guidance")

@st.cache_resource(show_spinner=False)
def get_sentiment_pipeline():
    return pipeline("sentiment-analysis")

sentiment_pipeline = get_sentiment_pipeline()

if "history" not in st.session_state:
    st.session_state.history = []
if "sentiments" not in st.session_state:
    st.session_state.sentiments = []
if "scores" not in st.session_state:
    st.session_state.scores = []

col1, col2 = st.columns(2)

with col1:
    st.subheader("User")
    user_input = st.text_input("Type your message:", key="user_input")
    send_button = st.button("Send")

with col2:
    st.subheader("Advisor")
    whisper = st.text_input("Whisper to the bot:", key="whisper_input")

def analyze_sentiment(text):
    try:
        result = sentiment_pipeline(text)[0]
        label = result["label"]
        score = result["score"]
        return label, score
    except Exception:
        return "NEUTRAL", 0.5

def get_sentiment_display(label, score):
    color = {
        "POSITIVE": "#d4edda",  # green
        "NEGATIVE": "#f8d7da",  # red
        "NEUTRAL": "#fff3cd"    # yellow
    }.get(label, "#f8f9fa")
    return f"""
        <div style="background-color:{color};border-radius:8px;padding:6px 12px;display:inline-block;">
        <b>{label.capitalize()}</b> (score: {score:.2f})
        </div>
    """

prompt = PromptTemplate.from_template("""
You are a helpful, warm customer support assistant.

You MUST follow any supervisor advice given below exactly and prioritize it in your reply.

Context:
- Past memory: User has had password trouble on mobile before.
- Supervisor advice (optional): {whisper}
- User sentiment: {sentiment_label} (score: {sentiment_score:.2f})

Recent conversation:
{history}

User: {user_input}
Assistant:
""")

if send_button and user_input.strip():
    label, score = analyze_sentiment(user_input)
    st.session_state.sentiments.append(label)
    st.session_state.scores.append(score)

    def get_whisper(_): return whisper
    def get_user_input(_): return user_input
    def get_history(_): return "\n".join(st.session_state.get("history", [])[-6:])
    def get_sentiment_label(_): return label
    def get_sentiment_score(_): return score

    pipeline_map = (
        RunnableMap({
            "whisper": get_whisper,
            "user_input": get_user_input,
            "history": get_history,
            "sentiment_label": get_sentiment_label,
            "sentiment_score": get_sentiment_score,
        })
        | prompt
        | llm
    )

    response = pipeline_map.invoke({})
    st.session_state.history.append(f"user: {user_input}")
    st.session_state.history.append(f"assistant: {response.content}")

# Display chat history with sentiment, and show warning for strong negative sentiment
for i, msg in enumerate(st.session_state.history):
    role, content = msg.split(": ", 1)
    with st.chat_message(role):
        st.markdown(content)
        if role == "user":
            label = st.session_state.sentiments[i // 2]
            score = st.session_state.scores[i // 2]
            st.markdown(get_sentiment_display(label, score), unsafe_allow_html=True)
            # Show red warning if strongly negative
            if label == "NEGATIVE" and score > 0.7:
                st.error("Strong negative sentiment detected. Advisor intervention is recommended.")