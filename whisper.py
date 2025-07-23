import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_google_genai import ChatGoogleGenerativeAI
from transformers import pipeline
from datetime import datetime

from graphiti_core import Graphiti
from graphiti_core.llm_client import LLMConfig
from graphiti_core.llm_client.gemini_client import GeminiClient
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig

# Load env variables
load_dotenv()
st.set_page_config(page_title="Gemini Whisper Bot", layout="wide")

def get_env_var(key):
    if hasattr(st, "secrets") and key in st.secrets:
        return st.secrets[key]
    return os.getenv(key)

neo4j_url = get_env_var("NEO4J_URL")
neo4j_user = get_env_var("NEO4J_USER")
neo4j_password = get_env_var("NEO4J_PASSWORD")
GOOGLE_API_KEY = get_env_var("GOOGLE_API_KEY")

@st.cache_resource(show_spinner=False)
def get_graphiti():
    if not all([neo4j_url, neo4j_user, neo4j_password, GOOGLE_API_KEY]):
        raise ValueError("One or more credentials are missing or empty!")
    llm_client = GeminiClient(
        config=LLMConfig(
            api_key=GOOGLE_API_KEY,
            model="gemini-2.0-flash"
        )
    )
    embedder = GeminiEmbedder(
        config=GeminiEmbedderConfig(
            api_key=GOOGLE_API_KEY,
            embedding_model="embedding-001"
        )
    )
    return Graphiti(
        neo4j_url,
        neo4j_user,
        neo4j_password,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=None,
    )

graphiti = get_graphiti()

@st.cache_resource(show_spinner=False)
def get_sentiment_pipeline():
    return pipeline("sentiment-analysis")
sentiment_pipeline = get_sentiment_pipeline()

st.title("Gemini Bot with Whispered Guidance")

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
        "POSITIVE": "#d4edda",
        "NEGATIVE": "#f8d7da",
        "NEUTRAL": "#fff3cd"
    }.get(label, "#f8f9fa")
    return f"""
        <div style="background-color:{color};border-radius:8px;padding:6px 12px;display:inline-block;">
        <b>{label.capitalize()}</b> (score: {score:.2f})
        </div>
    """

def store_message(role, content):
    # You can set group_id to a user/session ID if you want, or leave as "default"
    graphiti.add_episode(
        name="Chatbot Message",
        episode_body=f"{role}: {content}",
        source="message",
        source_description="Chatbot",
        reference_time=datetime.now(),
        group_id="default"
    )

def retrieve_recent_history(n=6):
    # Returns latest episodes for this group
    episodes = graphiti.retrieve_episodes(datetime.now(), last_n=n, group_ids=["default"])
    messages = []
    for ep in reversed(episodes):  # Oldest first
        for line in ep.content.split("\n"):
            if ": " in line:
                messages.append(line)
    return "\n".join(messages)

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

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

    store_message("user", user_input)
    full_history = retrieve_recent_history(n=6)

    def get_whisper(_): return whisper
    def get_user_input(_): return user_input
    def get_history(_): return full_history
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

    store_message("assistant", response.content)
    st.session_state.history.append(f"user: {user_input}")
    st.session_state.history.append(f"assistant: {response.content}")

for i, msg in enumerate(st.session_state.history):
    role, content = msg.split(": ", 1)
    with st.chat_message(role):
        st.markdown(content)
        if role == "user":
            label = st.session_state.sentiments[i // 2]
            score = st.session_state.scores[i // 2]
            st.markdown(get_sentiment_display(label, score), unsafe_allow_html=True)
            if label == "NEGATIVE" and score > 0.7:
                st.error("Strong negative sentiment detected. Advisor intervention is recommended.")

