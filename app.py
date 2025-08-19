# app.py
# 🚀 Social Media Sentiment Analyzer — improved
# - Robust Hugging Face scoring (fixes all-NEUTRAL/0.00)
# - Works in Demo without keys
# - Optional Gemini insights (auto-fallback, never blocks UI)
# - Optional Twitter (if TWITTER_BEARER_TOKEN available)
# - Neon dark UI, responsive charts, safe caching

import os
import time
import random
import re
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests

# -------------------------
# Optional deps (tweepy, torch, transformers)
# -------------------------
try:
    import tweepy  # type: ignore
except Exception:  # pragma: no cover
    tweepy = None

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    class _TorchStub:
        @staticmethod
        def cuda():
            class _Cuda:
                @staticmethod
                def is_available():
                    return False
            return _Cuda()
    torch = _TorchStub()

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        pipeline as hf_pipeline,
    )  # type: ignore
except Exception as e:  # pragma: no cover
    st.error("Transformers not installed. Add `transformers` to requirements.txt.")
    raise

# ==========================
# Page config & CSS
# ==========================
st.set_page_config(
    page_title="🚀 Social Media Sentiment Analyzer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');

:root {
  --fg: #ffffff;
  --muted: #b9b9b9;
  --glass: rgba(255,255,255,0.06);
  --border: rgba(255,255,255,0.10);
  --pos: #00ff41;
  --neu: #ffd700;
  --neg: #ff4757;
}

.stApp {
  background: radial-gradient(1200px 600px at 20% -10%, #1b1b2f 0%, #0b0b13 45%),
              radial-gradient(900px 500px at 120% 10%, #182848 0%, transparent 60%),
              linear-gradient(180deg, #0a0a0a 0%, #0a0a0a 100%);
  color: var(--fg);
}

.main-header {
  font-family: 'Orbitron', monospace;
  font-size: 2.6rem;
  font-weight: 900;
  text-align: center;
  background: linear-gradient(45deg, #00f5ff, #ff00ff, #00ff41);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: .4rem;
  text-shadow: 0 0 20px rgba(0,245,255,0.08);
}
.small { opacity: 0.85; font-size: 0.95rem; text-align:center; margin-bottom: 10px; }
.card { background: var(--glass); border: 1px solid var(--border); border-radius: 14px; padding: 12px; }
.badge { display:inline-block; padding:6px 12px; border-radius:18px; font-weight:700; }
.badge-live { background: linear-gradient(45deg,#00ff41,#00cc33); color:#000; }
.badge-demo { background: linear-gradient(45deg,#ff6b35,#f7941d); color:#000; }
.badge-stop { background: #666; color:#fff; }
.kpi { background: var(--glass); border:1px solid var(--border); border-radius: 12px; padding: 10px 12px; }
.kpi h3 { margin: 0 0 4px 0; font-size: 0.9rem; font-weight: 700; color: var(--muted); }
.kpi .v { font-size: 1.4rem; font-weight: 800; }
.post { background: var(--glass); border:1px solid var(--border); border-radius: 12px; padding: 12px; margin: 8px 0; }
</style>
""",
    unsafe_allow_html=True,
)

# ==========================
# Utility / session helpers
# ==========================

def init_session_state():
    defaults = {
        "running": False,
        "posts_df": pd.DataFrame(columns=["timestamp", "text", "sentiment", "score", "source", "hashtag"]),
        "last_fetch": datetime.now(timezone.utc),
        "tweet_count": 0,
        "last_error": None,
        "last_error_when": None,
        "gemini_cache": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def set_last_error(exc: Exception, where: str = ""):
    msg = f"{where}: {type(exc).__name__}: {str(exc)}"
    st.session_state.last_error = msg
    st.session_state.last_error_when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    st.error(f"Error — {msg}")


init_session_state()

# ==========================
# Model loading (robust)
# ==========================

MODEL_NAME = os.getenv("HF_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest")

@st.cache_resource(show_spinner=True)
def load_hf_pipeline():
    """Build a transformers pipeline that returns full class scores
    and exposes id2label for correct mapping. Fixes NEUTRAL/0.00 bug.
    """
    device = 0 if hasattr(torch, "cuda") and torch.cuda.is_available() else -1
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    pipe = hf_pipeline(
        "text-classification",
        model=mdl,
        tokenizer=tok,
        truncation=True,
        return_all_scores=True,  # IMPORTANT — get all classes
        device=device,
    )
    id2label = mdl.config.id2label
    return pipe, id2label


# ==========================
# Text cleaning utilities
# ==========================
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")


def clean_text(text: str) -> str:
    if not text:
        return ""
    t = URL_RE.sub("", text)
    t = MENTION_RE.sub("", t)
    t = t.replace("\n", " ").strip()
    t = " ".join(t.split())
    return t


# ==========================
# Sentiment helper (robust mapping + softmax guard)
# ==========================

import numpy as _np

def _softmax(x):
    x = _np.array(x, dtype=_np.float64)
    x = x - _np.max(x)
    ex = _np.exp(x)
    return (ex / _np.sum(ex)).tolist()


def classify_texts(texts: List[str], pipe, id2label: Dict[int, str]) -> List[Dict[str, Any]]:
    """Return list of {label, score, probs} per text.
    - Uses id2label to map LABEL_i → correct names
    - Applies softmax if scores don't sum ~1.0
    - Picks argmax class
    """
    if not texts:
        return []
    raw = pipe(texts)  # list[list[{label, score}...]]
    out = []
    for item in raw:
        labs, probs = [], []
        for d in item:
            lab = d.get("label", "").upper()
            if lab.startswith("LABEL_"):
                try:
                    idx = int(lab.split("_")[1])
                    lab = id2label.get(idx, lab)
                except Exception:
                    pass
            lab = lab.lower()
            labs.append(lab)
            probs.append(float(d.get("score", 0.0)))
        s = sum(probs)
        if not (0.99 <= s <= 1.01):  # guard if logits leaked
            probs = _softmax(probs)
        by = {labs[i]: probs[i] for i in range(len(labs))}
        neg = by.get("negative", by.get("neg", 0.0))
        neu = by.get("neutral", by.get("neu", 0.0))
        pos = by.get("positive", by.get("pos", 0.0))
        triplet = {"negative": neg, "neutral": neu, "positive": pos}
        top = max(triplet, key=triplet.get)
        out.append({
            "label": top.upper(),
            "score": float(triplet[top]),
            "probs": triplet,
        })
    return out


def numeric_score(row: Dict[str, Any]) -> float:
    lab = (row.get("label") or "").upper()
    p = row.get("probs", {})
    if lab == "POSITIVE":
        return float(p.get("positive", row.get("score", 0.0)))
    if lab == "NEGATIVE":
        return -float(p.get("negative", row.get("score", 0.0)))
    return 0.0


# ==========================
# Demo synthetic posts
# ==========================

def demo_post(tag: str) -> str:
    templates = [
        f"Just tried {tag} — impressed! 🔥",
        f"Not sure about #{tag}, kinda overhyped 🤔",
        f"Absolutely loving {tag}! Best decision ❤️",
        f"{tag} disappointed me. Expected more. 😞",
        f"The {tag} community is so supportive! 🌟",
        f"Why is everyone talking about {tag}? It's okay I guess.",
        f"{tag} changed my workflow. Mind blown 🤯",
        f"Can't stop thinking about {tag} — life improved 🙏",
        f"Tried {tag} — neutral feelings overall.",
        f"Big updates for {tag} — promising ✨",
        f"I hate the new {tag} change. Very buggy 😤",
        f"Team {tag} all the way! Who's with me? 💪",
    ]
    return random.choice(templates)


# ==========================
# Twitter helper (optional)
# ==========================

def setup_twitter_client() -> Optional["tweepy.Client"]:
    if not tweepy:
        return None
    token = os.getenv("TWITTER_BEARER_TOKEN")
    if not token:
        return None
    try:
        client = tweepy.Client(bearer_token=token, wait_on_rate_limit=True)
        return client
    except Exception as e:
        set_last_error(e, "setup_twitter_client")
        return None


def fetch_tweets(client: "tweepy.Client", hashtag: str, n: int = 10) -> List[str]:
    if not client:
        return []
    try:
        q = f"#{hashtag} -is:retweet lang:en"
        resp = client.search_recent_tweets(query=q, max_results=min(n, 100), tweet_fields=["created_at", "text"])
        texts = [t.text for t in (getattr(resp, "data", []) or [])]
        return texts
    except Exception as e:
        set_last_error(e, "fetch_tweets")
        return []


# ==========================
# Gemini helper (optional)
# ==========================
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "").strip()


def gemini_insights(posts: pd.DataFrame, hashtag: str) -> Optional[str]:
    if not GEMINI_KEY:
        return None
    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        headers = {"Content-Type": "application/json", "X-goog-api-key": GEMINI_KEY}
        sample = posts.tail(20)["text"].tolist()
        prompt = (
            "You are a product analyst. In <= 40 words, give 3 crisp bullet insights about overall sentiment and themes for posts "
            f"about #{hashtag}. No emojis, no hashtags.\n\nSample posts (newest last):\n" + "\n".join(sample)
        )
        payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.4, "maxOutputTokens": 120}}
        r = requests.post(url, headers=headers, json=payload, timeout=15)
        if r.status_code != 200:
            set_last_error(RuntimeError(f"Gemini {r.status_code}: {r.text[:200]}"), "gemini_insights")
            return None
        data = r.json()
        return (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
            .strip()
            or None
        )
    except Exception as e:
        set_last_error(e, "gemini_insights")
        return None


# ==========================
# Main App
# ==========================

def main():
    st.markdown('<div class="main-header">🚀 Social Media Sentiment Analyzer</div>', unsafe_allow_html=True)
    st.markdown(
        "<div class='small'>Hugging Face sentiment + optional Gemini insights • Demo works without keys</div>",
        unsafe_allow_html=True,
    )

    # Sidebar
    st.sidebar.title("🎛️ Control Panel")
    hashtag = st.sidebar.text_input("🏷️ Hashtag (no #)", value="AI", max_chars=64)
    window_size = st.sidebar.slider("📊 Rolling window size (posts)", 10, 500, 100, step=10)
    refresh = st.sidebar.slider("⏱️ Refresh interval (sec)", 5, 60, 10, 1)

    twitter_client = setup_twitter_client()
    live_available = twitter_client is not None
    mode_opts = ["Auto (Prefer Live)", "Force Live", "Force Demo"] if live_available else ["Demo Mode Only"]
    mode = st.sidebar.selectbox("🔧 Mode", mode_opts, index=0)

    use_gemini = st.sidebar.checkbox("Enable Gemini Insights", value=False)
    if use_gemini and not GEMINI_KEY:
        st.sidebar.info("Set GEMINI_API_KEY in your environment to enable insights.")

    c1, c2 = st.sidebar.columns(2)
    start = c1.button("▶ Start")
    stop = c2.button("⏹ Stop")
    if start:
        st.session_state.running = True
        st.balloons()
    if stop:
        st.session_state.running = False

    if st.sidebar.button("🗑 Clear data"):
        st.session_state.posts_df = pd.DataFrame(columns=["timestamp", "text", "sentiment", "score", "source", "hashtag"])
        st.session_state.tweet_count = 0
        st.session_state.last_fetch = datetime.now(timezone.utc)
        st.session_state.gemini_cache = None
        st.experimental_rerun()

    # Status badge
    if st.session_state.running:
        if (mode == "Force Demo") or (mode == "Auto (Prefer Live)" and not live_available):
            st.sidebar.markdown('<span class="badge badge-demo">🎭 DEMO MODE</span>', unsafe_allow_html=True)
        else:
            st.sidebar.markdown('<span class="badge badge-live">📡 LIVE MODE</span>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<span class="badge badge-stop">⏸ STOPPED</span>', unsafe_allow_html=True)

    # Last error panel
    if st.session_state.last_error:
        with st.expander("⚠️ Last Error (click to inspect)"):
            st.write(st.session_state.last_error)
            st.write("When:", st.session_state.last_error_when)

    # Load model once
    try:
        (pipe, id2label) = load_hf_pipeline()
    except Exception as e:
        set_last_error(e, "load_hf_pipeline")
        pipe, id2label = None, {}

    # Fetch cycle
    def fetch_cycle():
        try:
            use_live = False
            if mode == "Force Live":
                use_live = True
            elif mode == "Force Demo":
                use_live = False
            else:  # Auto
                use_live = live_available

            texts: List[str] = []
            source = "demo"
            tag = hashtag
            if use_live and twitter_client:
                texts = fetch_tweets(twitter_client, tag, n=6) or []
                source = "twitter" if texts else "demo"
            if not texts:
                texts = [demo_post(tag) for _ in range(4)]
                source = "demo"

            if not pipe:
                # If model not loaded, still push rows with neutral 0.0 so UI runs
                rows = [{
                    "timestamp": datetime.now(timezone.utc),
                    "text": t,
                    "sentiment": "NEUTRAL",
                    "score": 0.0,
                    "source": source,
                    "hashtag": tag,
                } for t in texts]
            else:
                cleaned = [clean_text(t) for t in texts]
                results = classify_texts(cleaned, pipe, id2label)
                rows = []
                for t, r in zip(texts, results):
                    ns = numeric_score(r)
                    rows.append({
                        "timestamp": datetime.now(timezone.utc),
                        "text": t,
                        "sentiment": r["label"],
                        "score": ns,
                        "source": source,
                        "hashtag": tag,
                    })

            if rows:
                new_df = pd.DataFrame(rows)
                df = pd.concat([st.session_state.posts_df, new_df], ignore_index=True)
                if len(df) > window_size:
                    df = df.tail(window_size).reset_index(drop=True)
                st.session_state.posts_df = df
                st.session_state.tweet_count += len(rows)
                st.session_state.last_fetch = datetime.now(timezone.utc)
        except Exception as e:
            set_last_error(e, "fetch_cycle")

    if st.session_state.running:
        delta = (datetime.now(timezone.utc) - st.session_state.last_fetch).total_seconds()
        if delta >= refresh:
            fetch_cycle()
            # Gemini async-ish cache update
            if use_gemini and len(st.session_state.posts_df) > 0:
                try:
                    insight = gemini_insights(st.session_state.posts_df, hashtag)
                    st.session_state.gemini_cache = insight
                except Exception as e:
                    set_last_error(e, "gemini_cache")
            st.experimental_rerun()

    # ================= UI: Dashboard =================
    st.markdown("## Dashboard")
    df = st.session_state.posts_df

    if len(df) == 0:
        st.markdown(
            """
            <div class="card" style="text-align:center;padding:28px;">
              <h3 style="margin:0 0 6px 0;">🎯 Ready to analyze!</h3>
              <div class="small">Set a hashtag, choose a mode, and click ▶ Start.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # KPIs
    total = len(df)
    counts = df["sentiment"].value_counts().to_dict()
    pos = counts.get("POSITIVE", 0)
    neu = counts.get("NEUTRAL", 0)
    neg = counts.get("NEGATIVE", 0)
    pos_pct = (pos / total) * 100
    neu_pct = (neu / total) * 100
    neg_pct = (neg / total) * 100

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"<div class='kpi'><h3>📈 Total Posts (window)</h3><div class='v'>{total:,}</div></div>", unsafe_allow_html=True)
    with k2:
        st.markdown(f"<div class='kpi'><h3>😊 Positive</h3><div class='v'>{pos_pct:.1f}%</div></div>", unsafe_allow_html=True)
    with k3:
        st.markdown(f"<div class='kpi'><h3>😐 Neutral</h3><div class='v'>{neu_pct:.1f}%</div></div>", unsafe_allow_html=True)
    with k4:
        st.markdown(f"<div class='kpi'><h3>☹️ Negative</h3><div class='v'>{neg_pct:.1f}%</div></div>", unsafe_allow_html=True)

    # Charts
    left, right = st.columns([2, 1])
    with left:
        st.markdown("### 📈 Sentiment Over Time")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df["score"],
            mode="lines+markers",
            name="score",
            line=dict(width=2),
            marker=dict(size=6),
        ))
        fig.add_hline(y=0, line_dash="dash")
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(range=[-1, 1], title="Sentiment Score"),
            xaxis=dict(title="Time"),
            height=420,
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("### 🍩 Sentiment Distribution")
        labels = ["Positive", "Neutral", "Negative"]
        values = [pos, neu, neg]
        fig2 = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.55)])
        fig2.update_traces(textinfo="percent+label")
        fig2.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=420,
            margin=dict(l=10, r=10, t=40, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Recent posts
    st.markdown("### 💬 Recent Posts")
    recent = df.tail(12).sort_values("timestamp", ascending=False)
    for _idx, row in recent.iterrows():
        color = "var(--pos)" if row["sentiment"] == "POSITIVE" else ("var(--neu)" if row["sentiment"] == "NEUTRAL" else "var(--neg)")
        ts = pd.to_datetime(row["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        st.markdown(
            f"""
            <div class='post' style='border-left:4px solid {color}'>
              <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:6px'>
                <div style='font-weight:800;color:{color}'>{row['sentiment']} ({row['score']:.2f})</div>
                <div style='color:#bbb;font-size:0.85rem'>{ts} • {row.get('source','')}</div>
              </div>
              <div style='color:#fff'>{row['text']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Gemini panel
    if use_gemini:
        st.markdown("### 🤖 Gemini Insights (optional)")
        if st.session_state.gemini_cache:
            st.info(st.session_state.gemini_cache)
        elif not GEMINI_KEY:
            st.warning("GEMINI_API_KEY not set. Add it to your environment to enable insights.")
        else:
            st.caption("Insights will appear after the next fetch cycle…")

    # Footer
    f1, f2 = st.columns([3, 1])
    with f1:
        st.markdown("<div class='small'>Tip: Demo mode is safe for showcases. Add TWITTER_BEARER_TOKEN / GEMINI_API_KEY for live & insights.</div>", unsafe_allow_html=True)
    with f2:
        st.markdown(
            f"<div class='small' style='text-align:right'>Total fetched: <b>{st.session_state.tweet_count:,}</b></div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        set_last_error(e, "main")
        st.exception(e)
