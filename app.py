import os
import re
import random
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timezone
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline as hf_pipeline
import tweepy

# ==========================
# Session State Init
# ==========================

def init_session_state():
    if "posts_df" not in st.session_state:
        st.session_state.posts_df = pd.DataFrame(columns=["text", "sentiment", "score", "timestamp", "source", "hashtag"])
    if "tweet_count" not in st.session_state:
        st.session_state.tweet_count = 0
    if "last_fetch" not in st.session_state:
        st.session_state.last_fetch = datetime.now(timezone.utc)
    if "last_error" not in st.session_state:
        st.session_state.last_error = None
        st.session_state.last_error_time = None
        st.session_state.last_error_loc = None

init_session_state()

# ==========================
# Error Handling
# ==========================

def set_last_error(e, loc="app"):
    st.session_state.last_error = str(e)
    st.session_state.last_error_time = datetime.now(timezone.utc)
    st.session_state.last_error_loc = loc

# ==========================
# Demo Posts
# ==========================

def demo_post(tag):
    demos = [
        f"Absolutely loving AI today! Best decision ever ❤️ #{tag}",
        f"Team AI all the way! Who's with me? 💪 #{tag}",
        f"I hate the new AI change. Very buggy 😤 #{tag}",
        f"Tried AI — neutral feelings overall. #{tag}",
        f"Not sure about #{tag}, feels overhyped... 🤔",
        f"AI disappointed me. Expected more. 😞 #{tag}"
    ]
    return random.choice(demos)

# ==========================
# Text Cleaning
# ==========================

def clean_text(txt: str) -> str:
    txt = re.sub(r"http\S+", "", txt)
    txt = re.sub(r"@\w+", "", txt)
    txt = re.sub(r"#\w+", "", txt)
    txt = txt.strip()
    return txt

# ==========================
# HuggingFace Pipeline
# ==========================

MODEL_NAME = os.getenv("HF_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest")

@st.cache_resource(show_spinner=True)
def load_hf_pipeline():
    device = 0 if torch.cuda.is_available() else -1
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    pipe = hf_pipeline(
        "text-classification",
        model=mdl,
        tokenizer=tok,
        truncation=True,
        top_k=None,   # ✅ safe replacement for return_all_scores
        device=device,
    )
    id2label = mdl.config.id2label
    return pipe, id2label

try:
    pipe, id2label = load_hf_pipeline()
except Exception as e:
    pipe, id2label = None, {}
    set_last_error(e, "load_hf")

# ==========================
# Classify Texts
# ==========================

def classify_texts(texts: List[str], pipe, id2label):
    results = []
    for t in texts:
        try:
            out = pipe(t, truncation=True)
            best = out[0][0] if isinstance(out[0], list) else out[0]
            label = id2label.get(best["label"], best["label"])
            results.append({"label": label, "score": float(best["score"])})
        except Exception as e:
            set_last_error(e, "classify")
            results.append({"label": "NEUTRAL", "score": 0.0})
    return results

def numeric_score(r):
    l = r["label"].upper()
    if "POS" in l:
        return r["score"]
    if "NEG" in l:
        return -r["score"]
    return 0.0

# ==========================
# Twitter Client (Optional)
# ==========================

def get_twitter_client():
    try:
        bearer = st.secrets.get("TWITTER_BEARER", None)
        if bearer:
            return tweepy.Client(bearer_token=bearer, wait_on_rate_limit=True)
    except Exception:
        pass
    return None

twitter_client = get_twitter_client()

def fetch_tweets(client, query, n=6):
    try:
        res = client.search_recent_tweets(query=query, max_results=n, tweet_fields=["text","created_at","lang"])
        if not res.data:
            return []
        return [t.text for t in res.data if t.lang == "en"]
    except Exception as e:
        set_last_error(e, "fetch_tweets")
        return []

# ==========================
# Fetch Cycle
# ==========================

def fetch_cycle(hashtag, mode, window_size):
    try:
        use_live = (mode == "Force Live") or (mode == "Auto" and twitter_client is not None)
        texts = []
        source = "demo"

        if use_live and twitter_client:
            texts = fetch_tweets(twitter_client, hashtag, n=6) or []
            source = "twitter" if texts else "demo"

        if not texts:
            texts = [demo_post(hashtag) for _ in range(4)]
            source = "demo"

        if not pipe:
            rows = [{"timestamp": datetime.now(timezone.utc), "text": t, "sentiment": "NEUTRAL", "score": 0.0, "source": source, "hashtag": hashtag} for t in texts]
        else:
            cleaned = [clean_text(t) for t in texts]
            results = classify_texts(cleaned, pipe, id2label)
            rows = []
            for t, r in zip(texts, results):
                ns = numeric_score(r)
                rows.append({"timestamp": datetime.now(timezone.utc), "text": t, "sentiment": r["label"], "score": ns, "source": source, "hashtag": hashtag})

        if rows:
            new_df = pd.DataFrame(rows)
            if not new_df.empty:
                df = pd.concat([st.session_state.posts_df, new_df], ignore_index=True)
                if len(df) > window_size:
                    df = df.tail(window_size).reset_index(drop=True)
                st.session_state.posts_df = df
                st.session_state.tweet_count += len(rows)
                st.session_state.last_fetch = datetime.now(timezone.utc)
    except Exception as e:
        set_last_error(e, "fetch_cycle")

# ==========================
# Dashboard
# ==========================

def dashboard():
    st.title("🚀 Social Media Sentiment Analyzer")
    st.caption("Hugging Face sentiment + optional Twitter/Gemini • Demo mode works without keys")

    st.sidebar.header("⚙️ Settings")
    hashtag = st.sidebar.text_input("Hashtag / Keyword", "AI")
    mode = st.sidebar.selectbox("Mode", ["Auto", "Force Demo", "Force Live"])
    window_size = st.sidebar.slider("Window size", 10, 200, 50)
    auto = st.sidebar.checkbox("Auto-refresh every 30s", True)

    st.markdown("### 📊 Stats")
    total = len(st.session_state.posts_df)
    pos = (st.session_state.posts_df["score"] > 0).mean() if total > 0 else 0
    neg = (st.session_state.posts_df["score"] < 0).mean() if total > 0 else 0
    neu = 1 - pos - neg if total > 0 else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Posts", total)
    k2.metric("😊 Positive", f"{pos*100:.1f}%")
    k3.metric("😐 Neutral", f"{neu*100:.1f}%")
    k4.metric("☹️ Negative", f"{neg*100:.1f}%")

    if total > 0:
        st.markdown("### 📈 Sentiment Over Time")
        st.plotly_chart(px.line(st.session_state.posts_df, x="timestamp", y="score", markers=True), use_container_width=True)

        st.markdown("### 🍩 Sentiment Distribution")
        st.plotly_chart(px.pie(st.session_state.posts_df, names="sentiment"), use_container_width=True)

        st.markdown("### 💬 Recent Posts")
        for _, row in st.session_state.posts_df.tail(10).iloc[::-1].iterrows():
            st.write(f"**{row['sentiment']} ({row['score']:.2f})** • {row['timestamp']} • {row['source']}")
            st.caption(row["text"])

    if st.session_state.last_error:
        with st.expander("⚠️ Last Error (click to inspect)"):
            st.error(f"{st.session_state.last_error}\n\nAt: {st.session_state.last_error_loc} • {st.session_state.last_error_time}")

    if auto or st.sidebar.button("🔄 Fetch Now"):
        fetch_cycle(hashtag, mode, window_size)
        st.rerun()

# ==========================
# Entrypoint
# ==========================

if __name__ == "__main__":
    try:
        dashboard()
    except Exception as e:
        set_last_error(e, "main")
