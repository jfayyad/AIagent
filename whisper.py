import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_google_genai import ChatGoogleGenerativeAI
from transformers import pipeline
import json

# Load environment variables
load_dotenv()

# --- Knowledge base loader ---
def load_knowledge_base(filepath="company.json"):
    with open(filepath, "r") as f:
        kb = json.load(f)
    kb_prompt = f"""
You are an expert customer support assistant for {kb['company_name']}.

Company Overview: {kb['overview']}
Core Products: {", ".join(kb["core_products"])}
Key Features: {", ".join(kb["key_features"])}
Sustainability: {", ".join(kb["sustainability"])}
Customer Experience: {", ".join(kb["customer_experience"])}
Mission: {kb["mission"]}
FAQs: {" ".join(kb["faqs"])}
Contact: Website: {kb["contact"]["website"]}, Email: {kb["contact"]["email"]}, Social: {", ".join(kb["contact"]["social"])}
"""
    return kb_prompt

# Load knowledge base at app startup
kb_prompt = load_knowledge_base()

# --- LLM setup ---
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

st.set_page_config(page_title="Gemini Whisper Bot", layout="wide")
st.title("Gemini Bot with Guidance")

# --- Improved Sentiment Analysis ---
@st.cache_resource(show_spinner=False)
def get_sentiment_pipeline():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    return pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)

sentiment_pipeline = get_sentiment_pipeline()

def analyze_sentiment(text):
    try:
        result = sentiment_pipeline(text)[0]
        label = result["label"].upper()  # 'POSITIVE', 'NEGATIVE', 'NEUTRAL'
        score = result["score"]
        # If the message is a question and classified as negative, check for negative words
        negative_words = ["bad", "worst", "angry", "problem", "issue", "disappointed", "upset", "frustrated"]
        is_question = "?" in text
        contains_neg = any(word in text.lower() for word in negative_words)
        if is_question and label == "NEGATIVE" and not contains_neg:
            label = "NEUTRAL"
            score = 0.5
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

# --- Prompt ---
prompt = PromptTemplate.from_template(f"""
{kb_prompt}

You MUST follow any supervisor advice given below exactly and prioritize it in your reply.

Context:
- Supervisor advice (optional): {{whisper}}
- User sentiment: {{sentiment_label}} (score: {{sentiment_score:.2f}})

Recent conversation:
{{history}}

User: {{user_input}}
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
