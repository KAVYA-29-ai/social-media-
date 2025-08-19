# app.py
import os
import time
import random
import re
from datetime import datetime
from typing import List, Optional

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import tweepy
from transformers import pipeline
import torch

# ==========================
# Page config & CSS
# ==========================
st.set_page_config(
    page_title="🚀 Social Media Sentiment Analyzer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom dark/neon CSS
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');

.stApp {
    background: linear-gradient(45deg, #0a0a0a, #1a1a2e, #16213e);
    background-size: 400% 400%;
    animation: gradientShift 12s ease infinite;
    color: #fff;
}
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.main-header {
    font-family: 'Orbitron', monospace;
    font-size: 2.6rem;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(45deg, #00f5ff, #ff00ff, #00ff41);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 1rem;
    text-shadow: 0 0 20px rgba(0,245,255,0.08);
}
.metric-card {
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 14px;
    margin: 6px;
    border: 1px solid rgba(255,255,255,0.06);
}
.status-badge { display:inline-block; padding:6px 12px; border-radius:18px; font-weight:700; }
.status-live { background: linear-gradient(45deg,#00ff41,#00cc33); color:#000; }
.status-demo { background: linear-gradient(45deg,#ff6b35,#f7941d); color:#000; }
.small { opacity:0.8; font-size:0.9rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ==========================
# Utility / session helpers
# ==========================
def init_session_state():
    """Initialize session state variables used across the app."""
    defaults = {
        "running": False,
        "posts_df": pd.DataFrame(columns=["timestamp", "text", "sentiment", "score", "hashtag", "source"]),
        "tweet_count": 0,
        "last_update": datetime.utcnow(),
        "last_error": None,
        "last_error_time": None,
        "gemini_cache": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def set_last_error(exc: Exception, where: str = ""):
    """Store last error in session state with readable message and timestamp."""
    msg = f"{where}: {type(exc).__name__}: {str(exc)}"
    st.session_state.last_error = msg
    st.session_state.last_error_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    # Also log to Streamlit's logger (helpful for spaces logs)
    st.error(f"Error — {msg}")

init_session_state()

# ==========================
# Model loading (cached)
# ==========================
@st.cache_resource
def load_sentiment_model():
    """Load the HF transformers sentiment pipeline with safe device selection."""
    try:
        device = 0 if torch.cuda.is_available() else -1
        # Use cardiffnlp twitter-roberta sentiment model (free)
        model = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=device,
            truncation=True,
            top_k=None,
        )
        return model
    except Exception as e:
        # Pass exception upstream
        raise RuntimeError(f"Failed to load sentiment model: {e}")

# ==========================
# Text cleaning utilities
# ==========================
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")

def clean_text(text: str) -> str:
    """Remove URLs, mentions and excess whitespace for model input."""
    if not text:
        return ""
    t = URL_RE.sub("", text)
    t = MENTION_RE.sub("", t)
    t = t.replace("\n", " ").strip()
    t = " ".join(t.split())
    return t

# ==========================
# Sentiment helper
# ==========================
LABEL_MAP = {
    # cardiffnlp uses LABEL_0, LABEL_1, LABEL_2 -> map accordingly
    "LABEL_0": "NEGATIVE",
    "LABEL_1": "NEUTRAL",
    "LABEL_2": "POSITIVE",
    "NEGATIVE": "NEGATIVE",
    "NEUTRAL": "NEUTRAL",
    "POSITIVE": "POSITIVE",
}

def analyze_sentiment(text: str, model) -> (str, float):
    """Return (sentiment_label, confidence_score). Handles model label formats robustly."""
    try:
        if model is None:
            raise RuntimeError("Sentiment model not available")
        cleaned = clean_text(text)
        if not cleaned:
            return "NEUTRAL", 0.0
        out = model(cleaned)[0]
        raw_label = out.get("label") or out.get("result") or ""
        score = float(out.get("score", 0.0))
        label = LABEL_MAP.get(raw_label.upper(), "NEUTRAL")
        return label, score
    except Exception as e:
        set_last_error(e, "analyze_sentiment")
        return "NEUTRAL", 0.0

# ==========================
# Demo synthetic stream
# ==========================
def generate_demo_post(hashtag: str) -> str:
    """Return a plausible demo social post for given hashtag."""
    try:
        templates = [
            f"Just tried the new {hashtag} experience — wow, I'm impressed! 🔥",
            f"Not sure about #{hashtag}, feels overhyped... 🤔",
            f"Absolutely loving {hashtag} today! Best decision ever ❤️",
            f"{hashtag} disappointed me. Expected more. 😞",
            f"The {hashtag} community is so supportive! 🌟",
            f"Why is everyone talking about {hashtag}? It's okay I guess.",
            f"{hashtag} changed my workflow completely. Mind blown 🤯",
            f"Can't stop thinking about {hashtag} — life improved 🙏",
            f"Tried {hashtag} — neutral feelings overall.",
            f"Big updates for {hashtag} — looks promising ✨",
            f"I hate the new {hashtag} change. Very buggy 😤",
            f"Team {hashtag} all the way! Who's with me? 💪",
            f"Confused about the {hashtag} hype. Explain? 📣",
            f"Perfect {hashtag} moment today! Feeling grateful 😊",
        ]
        return random.choice(templates)
    except Exception as e:
        set_last_error(e, "generate_demo_post")
        # Fallback minimal text
        return f"{hashtag} demo post"

# ==========================
# Twitter helper (optional)
# ==========================
def setup_twitter_client() -> Optional[tweepy.Client]:
    """Return a tweepy.Client if TWITTER_BEARER_TOKEN is set and works, else None."""
    bearer = os.getenv("TWITTER_BEARER_TOKEN")
    if not bearer:
        return None
    try:
        client = tweepy.Client(bearer_token=bearer, wait_on_rate_limit=True)
        # test call
        try:
            client.get_me()
        except Exception:
            # some tokens might not allow get_me; ignore if search works later
            pass
        return client
    except Exception as e:
        set_last_error(e, "setup_twitter_client")
        return None

def fetch_tweets(client: tweepy.Client, hashtag: str, max_results: int = 10) -> List[str]:
    """Fetch recent tweets for a hashtag. Returns list of texts. Handles errors gracefully."""
    try:
        if client is None:
            return []
        query = f"#{hashtag} -is:retweet lang:en"
        resp = client.search_recent_tweets(query=query, max_results=min(max_results, 100),
                                          tweet_fields=["created_at", "text"])
        tweets = []
        if resp and getattr(resp, "data", None):
            for t in resp.data:
                tweets.append(t.text)
        return tweets
    except Exception as e:
        set_last_error(e, "fetch_tweets")
        return []

# ==========================
# Gemini (Google AI Studio) helper
# ==========================
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

def gemini_generate(prompt: str, timeout: int = 20) -> str:
    """Call Google Generative Language API (Gemini). Returns text or readable error message."""
    if not GEMINI_KEY:
        return "⚠️ No GEMINI_API_KEY set. Enable in your Space secrets to use Gemini."
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {"Content-Type": "application/json", "X-goog-api-key": GEMINI_KEY}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if resp.status_code != 200:
            # Save error for debugging
            set_last_error(RuntimeError(f"Gemini status {resp.status_code}: {resp.text}"), "gemini_generate")
            return f"⚠️ Gemini API error (status {resp.status_code}). See 'Last Error' panel."
        data = resp.json()
        # Defensive access: follow structure generative responses use
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return text
        except Exception:
            # Fallback to top-level textual fields if present
            txt = str(data)
            set_last_error(RuntimeError("Unexpected Gemini response shape"), "gemini_generate")
            return f"⚠️ Gemini response parsing failed. Raw: {txt[:500]}"
    except Exception as e:
        set_last_error(e, "gemini_generate")
        return "⚠️ Gemini request failed. See 'Last Error' panel."

# ==========================
# App main UI & logic
# ==========================
def main():
    # Header
    st.markdown('<div class="main-header">🚀 Social Media Sentiment Analyzer</div>', unsafe_allow_html=True)
    st.markdown("<div class='small' style='text-align:center;margin-bottom:18px;'>Hugging Face sentiment + optional Gemini insights • Demo mode works without keys</div>", unsafe_allow_html=True)

    # Load model with clear error feedback
    model = None
    try:
        with st.spinner("Loading sentiment model (cached)..."):
            model = load_sentiment_model()
    except Exception as e:
        set_last_error(e, "load_sentiment_model")
        st.sidebar.error("Failed to load sentiment model. Check logs. App will run in demo-only mode.")
        model = None

    # Sidebar controls
    st.sidebar.title("🎛️ Control Panel")
    twitter_client = setup_twitter_client()
    has_twitter = twitter_client is not None
    mode_options = ["Auto (Prefer Live)", "Force Live", "Force Demo"] if has_twitter else ["Demo Mode Only"]
    mode = st.sidebar.selectbox("🔧 Mode", mode_options, index=0)
    hashtag = st.sidebar.text_input("🏷️ Hashtag (no #)", value="AI", max_chars=64)
    refresh_interval = st.sidebar.slider("⏱️ Refresh interval (seconds)", min_value=5, max_value=60, value=10, step=1)
    window_size = st.sidebar.slider("📊 Rolling window size (posts)", min_value=10, max_value=500, value=100, step=10)

    # Gemini toggle
    gemini_enabled = st.sidebar.checkbox("Enable Gemini Insights (optional)", value=False)
    if gemini_enabled and not GEMINI_KEY:
        st.sidebar.info("Add GEMINI_API_KEY to Space secrets to enable Gemini.")

    # Controls
    c1, c2 = st.sidebar.columns(2)
    if c1.button("▶ Start"):
        st.session_state.running = True
        st.balloons()
    if c2.button("⏹ Stop"):
        st.session_state.running = False

    if st.sidebar.button("🗑 Clear data"):
        st.session_state.posts_df = pd.DataFrame(columns=["timestamp", "text", "sentiment", "score", "hashtag", "source"])
        st.session_state.tweet_count = 0
        st.session_state.last_update = datetime.utcnow()
        st.experimental_rerun()

    # Status
    if st.session_state.running:
        if mode == "Force Demo" or (mode == "Auto (Prefer Live)" and not has_twitter):
            st.sidebar.markdown('<div class="status-badge status-demo">🎭 DEMO MODE</div>', unsafe_allow_html=True)
        else:
            st.sidebar.markdown('<div class="status-badge status-live">📡 LIVE MODE</div>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<div class="status-badge" style="background:#666;color:#fff;">⏸ STOPPED</div>', unsafe_allow_html=True)

    # Display last error (if any)
    if st.session_state.last_error:
        with st.expander("⚠️ Last Error (click to inspect)"):
            st.write(st.session_state.last_error)
            st.write("Occurred at:", st.session_state.last_error_time)

    # Main content area: KPIs and charts
    st.markdown("## Dashboard")
    df = st.session_state.posts_df

    # If running, fetch posts periodically
    def fetch_and_process_posts():
        """Fetch posts from live or demo, analyze sentiment, and append to session state."""
        try:
            use_live = False
            if mode == "Force Live":
                use_live = True
            elif mode == "Force Demo":
                use_live = False
            else:  # Auto
                use_live = has_twitter

            posts = []
            source_tag = "demo"
            if use_live and twitter_client:
                fetched = fetch_tweets(twitter_client, hashtag, max_results=5)
                if fetched:
                    posts = fetched
                    source_tag = "twitter"
                else:
                    # fallback to demo if live fetch returns none
                    posts = [generate_demo_post(hashtag) for _ in range(3)]
                    source_tag = "demo"
            else:
                posts = [generate_demo_post(hashtag) for _ in range(3)]
                source_tag = "demo"

            rows = []
            for p in posts:
                try:
                    label, confidence = analyze_sentiment(p, model)
                except Exception as e:
                    set_last_error(e, "per-post analyze_sentiment")
                    label, confidence = "NEUTRAL", 0.0
                score = confidence if label == "POSITIVE" else (-confidence if label == "NEGATIVE" else 0.0)
                rows.append({
                    "timestamp": datetime.utcnow(),
                    "text": p,
                    "sentiment": label,
                    "score": score,
                    "hashtag": hashtag,
                    "source": source_tag,
                })

            if rows:
                new_df = pd.DataFrame(rows)
                combined = pd.concat([st.session_state.posts_df, new_df], ignore_index=True)
                # Trim to rolling window
                if len(combined) > window_size:
                    combined = combined.tail(window_size).reset_index(drop=True)
                st.session_state.posts_df = combined
                st.session_state.tweet_count += len(new_df)
                st.session_state.last_update = datetime.utcnow()
        except Exception as e:
            set_last_error(e, "fetch_and_process_posts")

    # If running, check timing and possibly fetch
    if st.session_state.running:
        time_since = (datetime.utcnow() - st.session_state.last_update).total_seconds()
        if time_since >= refresh_interval:
            fetch_and_process_posts()
            # After fetching, optionally request Gemini summary if enabled
            if gemini_enabled and len(st.session_state.posts_df) > 0:
                try:
                    # Create condensed prompt (last N small posts)
                    tail_texts = st.session_state.posts_df.tail(20)["text"].tolist()
                    joined = "\n".join(tail_texts)
                    prompt = f"Summarize overall sentiment and key themes for posts about #{hashtag}:\n\n{joined}"
                    gemini_result = gemini_generate(prompt)
                    st.session_state.gemini_cache = gemini_result
                except Exception as e:
                    set_last_error(e, "gemini_integration")
            # Rerun to update UI immediately
            st.experimental_rerun()

    # KPI metrics
    if len(df) > 0:
        total = len(df)
        counts = df["sentiment"].value_counts().to_dict()
        pos_pct = (counts.get("POSITIVE", 0) / total) * 100
        neu_pct = (counts.get("NEUTRAL", 0) / total) * 100
        neg_pct = (counts.get("NEGATIVE", 0) / total) * 100

        # Layout metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("📈 Total Posts (window)", f"{total:,}")
        m2.metric("😊 Positive", f"{pos_pct:.1f}%")
        m3.metric("😐 Neutral", f"{neu_pct:.1f}%")
        m4.metric("☹️ Negative", f"{neg_pct:.1f}%")

        # Charts
        left_col, right_col = st.columns([2, 1])
        with left_col:
            st.markdown("### 📈 Sentiment Over Time")
            if total > 1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df["timestamp"],
                    y=df["score"],
                    mode="lines+markers",
                    name="score",
                    line=dict(color="#00f5ff", width=2),
                    marker=dict(size=6),
                ))
                fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)")
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(range=[-1, 1], title="Sentiment Score"),
                    xaxis=dict(title="Time"),
                    height=420,
                    showlegend=False,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data yet to show time series. Start streaming to populate data.")

        with right_col:
            st.markdown("### 🍩 Sentiment Distribution")
            labels = ["Positive", "Neutral", "Negative"]
            vals = [pos_pct, neu_pct, neg_pct]
            fig2 = go.Figure(data=[go.Pie(labels=labels, values=vals, hole=0.6)])
            fig2.update_traces(textinfo="percent+label")
            fig2.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=420,
                margin=dict(l=10, r=10, t=40, b=10),
                showlegend=False
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Recent posts
        st.markdown("### 💬 Recent Posts")
        recent = df.tail(10).sort_values("timestamp", ascending=False)
        for _, row in recent.iterrows():
            color = "#00ff41" if row["sentiment"] == "POSITIVE" else ("#ffd700" if row["sentiment"] == "NEUTRAL" else "#ff4757")
            ts = row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            text_preview = row["text"]
            st.markdown(
                f"""
                <div style="background:rgba(255,255,255,0.03);border-radius:10px;padding:12px;margin:8px 0;border-left:4px solid {color}">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
                        <div style="font-weight:700;color:{color}">{row['sentiment']} ({row['score']:.2f})</div>
                        <div style="color:#bbb;font-size:0.85rem">{ts} • {row.get('source','')}</div>
                    </div>
                    <div style="color:#fff;">{text_preview}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    else:
        # Empty state prompt
        st.markdown(
            """
            <div style="text-align:center;padding:40px;">
                <h3 style="color:#00f5ff">🎯 Ready to analyze!</h3>
                <p style="color:#ccc">Configure hashtag, choose a mode, and click ▶ Start to stream demo or live posts.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Gemini insights panel (cached result displayed)
    if gemini_enabled:
        st.markdown("### 🤖 Gemini Insights (optional)")
        if st.session_state.gemini_cache:
            st.info(st.session_state.gemini_cache)
        elif not GEMINI_KEY:
            st.warning("GEMINI_API_KEY is not set. Add it to Space secrets to enable insights.")
        elif len(st.session_state.posts_df) == 0:
            st.write("No posts yet — start streaming to get Gemini insights.")
        else:
            st.write("Gemini will generate a short summary after the next fetch interval.")

    # Footer debug / controls
    footer_col1, footer_col2 = st.columns([3, 1])
    with footer_col1:
        st.markdown("<div class='small'>Tip: Demo mode is safe for classroom demos. Add TWITTER_BEARER_TOKEN and/or GEMINI_API_KEY as Space secrets to enable live/insights.</div>", unsafe_allow_html=True)
    with footer_col2:
        st.markdown("<div class='small' style='text-align:right'>Total fetched: <b>{:,}</b></div>".format(st.session_state.tweet_count), unsafe_allow_html=True)

# Entrypoint
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Last-resort error capture
        set_last_error(e, "main")
        st.exception(e)
