import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_google_genai import ChatGoogleGenerativeAI
from transformers import pipeline
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    filename="llm_debug.log",
    filemode="a",
    format="%(asctime)s %(levelname)s: %(message)s"
)
# ---- Always initialize session state ----
for var, val in [
    ("history", []),
    ("sentiments", []),
    ("scores", []),
    ("awaiting_supervisor", False),
    ("pending_user_input", ""),
    ("supervisor_input", ""),
    ("clear_user_input", False),
    ("clear_supervisor_input", False),
]:
    if var not in st.session_state:
        st.session_state[var] = val

# ---- Load company KB ----
load_dotenv()
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
kb_prompt = load_knowledge_base()

# ---- LLM setup ----
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

# ---- Sentiment ----
@st.cache_resource(show_spinner=False)
def get_sentiment_pipeline():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    return pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)
sentiment_pipeline = get_sentiment_pipeline()
def analyze_sentiment(text):
    try:
        result = sentiment_pipeline(text)[0]
        label = result["label"].upper()
        score = result["score"]
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

# ---- Certainty classifier ----
certainty_prompt = PromptTemplate.from_template("""
You are an expert at classifying customer support bot answers.

Instruction:
Given the following answer from a bot, decide if it is confident and provides helpful information ("CONFIDENT"),
or if it is uncertain, evasive, or cannot provide a real answer ("UNCERTAIN").
Reply with only one word: CONFIDENT or UNCERTAIN.

Bot answer:
"{bot_reply}"
""")
def classify_certainty(reply_text):
    cert_map = RunnableMap({"bot_reply": lambda _: reply_text})
    certainty_chain = cert_map | certainty_prompt | llm
    certainty_result = certainty_chain.invoke({})
    certainty = certainty_result.content.strip().upper()
    logging.info(f"Uncertainty LLM input: {reply_text}")
    logging.info(f"Uncertainty LLM output: {certainty}")
    return certainty

# ---- UI ----
st.set_page_config(page_title="Little Sleepies Bot", layout="wide")
st.title("Gemini Bot: Supervisor Guidance and Whisper")

col1, col2 = st.columns(2)

# --- Clear input fields BEFORE rendering widgets if flags are set ---
if st.session_state.clear_user_input:
    st.session_state["user_input"] = ""
    st.session_state.clear_user_input = False

if st.session_state.clear_supervisor_input:
    st.session_state["supervisor_input"] = ""
    st.session_state.clear_supervisor_input = False

with col1:
    st.subheader("User")
    user_input = st.text_input("Type your message:", key="user_input")
    send_button = st.button("Send", key="user_send")

with col2:
    st.subheader("Advisor")
    supervisor_input = st.text_input(
        "Whisper (normal) or Supervisor Advice (if needed):",
        key="supervisor_input",
        value=st.session_state.supervisor_input
    )
    supervisor_send = st.button("Send Supervisor Advice", key="supervisor_send")

# ---- Main prompt ----
main_prompt = PromptTemplate.from_template(f"""
{kb_prompt}

You are NOT a chatbot or virtual assistant; you represent Little Sleepies and must always answer as a company staff member, not as an AI. You are knowledgeable about all store policies, products, and details in the knowledge base.

IMPORTANT: The supervisor advice provided below is FACTUALLY CORRECT and MUST be included in your reply. If it is a short phrase, expand it naturally into a helpful sentence for the customer. Do NOT ignore, contradict, or hedge about the supervisor advice.

Supervisor advice: {{whisper}}

Context:
- User sentiment: {{sentiment_label}} (score: {{sentiment_score:.2f}})
- Recent conversation: 
{{history}}

User: {{user_input}}
Assistant:
""")


# ---- User sends a message ----
if send_button and user_input.strip():
    label, score = analyze_sentiment(user_input)
    st.session_state.sentiments.append(label)
    st.session_state.scores.append(score)

    # Advisor whisper (applies only to this turn)
    whisper_to_pass = supervisor_input if supervisor_input else ""
    def get_whisper(_): return whisper_to_pass
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
        | main_prompt
        | llm
    )

    response = pipeline_map.invoke({})
    response_text = response.content.strip()

    certainty = classify_certainty(response_text)
    st.session_state.history.append(f"user: {user_input}")

    if certainty == "UNCERTAIN":
        st.session_state.history.append(
            "assistant: Hold on, let me check with a supervisor before answering your question."
        )
        st.session_state.history.append(
            "supervisor_flag: Supervisor attention required for this user question."
        )
        st.session_state.awaiting_supervisor = True
        st.session_state.pending_user_input = user_input
    else:
        st.session_state.history.append(f"assistant: {response_text}")
        st.session_state.awaiting_supervisor = False
        st.session_state.pending_user_input = ""

    # Clear both fields after one use
    st.session_state.clear_user_input = True
    st.session_state.clear_supervisor_input = True
    st.rerun()

# ---- Supervisor sends advice in escalation mode ----
if st.session_state.awaiting_supervisor and supervisor_send and supervisor_input.strip():
    pending_input = st.session_state.get("pending_user_input", "")
    if not pending_input:
        st.warning("No pending user input available for supervisor escalation. Please wait for the user's question.")
    else:
        label, score = analyze_sentiment(pending_input)
        def get_whisper(_): return supervisor_input
        def get_user_input(_): return pending_input
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
            | main_prompt
            | llm
        )
        response = pipeline_map.invoke({})
        response_text = response.content.strip()
        st.session_state.history.append(f"assistant: {response_text}")
        st.session_state.awaiting_supervisor = False
        st.session_state.pending_user_input = ""
        # Clear supervisor input after one use
        st.session_state.clear_supervisor_input = True
        st.rerun()

# ---- Display chat (newest at top!) ----
user_count = len(st.session_state.sentiments) - 1
for msg in reversed(st.session_state.history):
    if msg.startswith("supervisor_flag:"):
        with st.chat_message("assistant"):
            st.error("🔴 Supervisor attention required for this question.")
        continue
    role, content = msg.split(": ", 1)
    with st.chat_message(role):
        st.markdown(content)
        if role == "user":
            if user_count >= 0:
                label = st.session_state.sentiments[user_count]
                score = st.session_state.scores[user_count]
                st.markdown(get_sentiment_display(label, score), unsafe_allow_html=True)
                if label == "NEGATIVE" and score > 0.7:
                    st.error("Strong negative sentiment detected. Advisor intervention is recommended.")
            user_count -= 1