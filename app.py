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

# ==========================
# Init Session State
# ==========================
if "posts_df" not in st.session_state:
    st.session_state.posts_df = pd.DataFrame(columns=["text","sentiment","score","timestamp","source","hashtag"])

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
# Clean Text
# ==========================
def clean_text(txt: str) -> str:
    txt = re.sub(r"http\S+", "", txt)
    txt = re.sub(r"@\w+", "", txt)
    txt = re.sub(r"#\w+", "", txt)
    return txt.strip()

# ==========================
# Load HF Pipeline
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
        top_k=None,    # ✅ latest replacement
        device=device,
    )
    id2label = mdl.config.id2label
    return pipe, id2label

try:
    pipe, id2label = load_hf_pipeline()
except Exception as e:
    pipe, id2label = None, {}

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
        except Exception:
            results.append({"label": "NEUTRAL", "score": 0.0})
    return results

def numeric_score(r):
    l = r["label"].upper()
    if "POS" in l: return r["score"]
    if "NEG" in l: return -r["score"]
    return 0.0

# ==========================
# Dashboard
# ==========================
def dashboard():
    st.title("🚀 Social Media Sentiment Analyzer")
    st.caption("Hugging Face sentiment • Demo mode works without keys")

    hashtag = st.sidebar.text_input("Hashtag / Keyword", "AI")
    window_size = st.sidebar.slider("Window size", 10, 200, 50)

    texts = [demo_post(hashtag) for _ in range(6)]

    if pipe:
        cleaned = [clean_text(t) for t in texts]
        results = classify_texts(cleaned, pipe, id2label)
        rows = []
        for t, r in zip(texts, results):
            rows.append({
                "timestamp": datetime.now(timezone.utc),
                "text": t,
                "sentiment": r["label"],
                "score": numeric_score(r),
                "source": "demo",
                "hashtag": hashtag
            })
        df = pd.DataFrame(rows)
        st.session_state.posts_df = pd.concat([st.session_state.posts_df, df], ignore_index=True)

    total = len(st.session_state.posts_df)
    pos = (st.session_state.posts_df["score"] > 0).mean() if total else 0
    neg = (st.session_state.posts_df["score"] < 0).mean() if total else 0
    neu = 1 - pos - neg if total else 0

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total", total)
    c2.metric("😊 Positive", f"{pos*100:.1f}%")
    c3.metric("😐 Neutral", f"{neu*100:.1f}%")
    c4.metric("☹️ Negative", f"{neg*100:.1f}%")

    if total > 0:
        st.plotly_chart(px.line(st.session_state.posts_df, x="timestamp", y="score"), use_container_width=True)
        st.plotly_chart(px.pie(st.session_state.posts_df, names="sentiment"), use_container_width=True)

        st.markdown("### Recent Posts")
        for _, row in st.session_state.posts_df.tail(5).iloc[::-1].iterrows():
            st.write(f"**{row['sentiment']} ({row['score']:.2f})** • {row['timestamp']}")
            st.caption(row["text"])

if __name__ == "__main__":
    dashboard()
