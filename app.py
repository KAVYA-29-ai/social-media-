import os
import random
import time
from typing import List, Dict

from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

import google.generativeai as genai
from transformers import pipeline

# -----------------------
# Flask setup
# -----------------------
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# -----------------------
# Config & Environment
# -----------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# Cap posts
MAX_POSTS = 50
DEFAULT_POSTS = 20

# -----------------------
# Sentiment Analyzer (HF)
# -----------------------
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
try:
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model=SENTIMENT_MODEL,
        device=-1  # CPU
    )
except Exception as e:
    print("Error initializing sentiment analyzer:", e)
    sentiment_analyzer = None

# -----------------------
# Helpers
# -----------------------
def normalize_count(n: int) -> int:
    try:
        n = int(n)
    except Exception:
        n = DEFAULT_POSTS
    n = max(1, min(MAX_POSTS, n))
    return n

def parse_sentiment(label: str, score: float) -> Dict[str, str]:
    if label.upper() == "POSITIVE":
        sentiment = "POSITIVE"
    elif label.upper() == "NEGATIVE":
        sentiment = "NEGATIVE"
    else:
        sentiment = "NEUTRAL"
    return {"sentiment": sentiment, "score": float(score)}

def compute_aggregate(rows: List[Dict]) -> Dict:
    pos = sum(1 for r in rows if r["sentiment"] == "POSITIVE")
    neg = sum(1 for r in rows if r["sentiment"] == "NEGATIVE")
    neu = sum(1 for r in rows if r["sentiment"] == "NEUTRAL")

    total = max(1, len(rows))
    pos_pct = round(100 * pos / total, 2)
    neg_pct = round(100 * neg / total, 2)
    neu_pct = round(100 * neu / total, 2)

    rolling = []
    score_map = {"POSITIVE": 1.0, "NEUTRAL": 0.5, "NEGATIVE": 0.0}
    alpha = 0.2
    ema = 0.5
    for r in rows:
        ema = alpha * score_map[r["sentiment"]] + (1 - alpha) * ema
        rolling.append(round(ema, 3))

    return {
        "counts": {"positive": pos, "negative": neg, "neutral": neu, "total": total},
        "percent": {"positive": pos_pct, "negative": neg_pct, "neutral": neu_pct},
        "rolling": rolling,
    }

# -----------------------
# Fallback posts
# -----------------------
FALLBACK_PATTERNS_POS = [
    "Absolutely loving {tag} right now! 🔥",
    "{tag} campaign is the best thing this season 🎉",
    "I love {tag}! It's amazing ❤️",
    "People are talking about {tag} everywhere 🌍",
    "Super excited about {tag} 🙌",
]
FALLBACK_PATTERNS_NEG = [
    "{tag} totally failed expectations 😠",
    "I'm disappointed with {tag} 💔",
    "{tag} needs serious improvements…",
    "Not impressed by {tag} this time 😕",
]
FALLBACK_PATTERNS_NEU = [
    "People are discussing {tag} a lot 🤔",
    "Not sure how I feel about {tag} yet…",
    "{tag} is trending — thoughts?",
    "Mixed opinions around {tag}.",
]

def make_fallback_posts(hashtag: str, n: int) -> List[str]:
    tag = hashtag if hashtag.startswith("#") else f"#{hashtag}"
    posts = []
    for _ in range(n):
        bucket = random.choices(
            [FALLBACK_PATTERNS_POS, FALLBACK_PATTERNS_NEU, FALLBACK_PATTERNS_NEG],
            weights=[0.4, 0.35, 0.25],
            k=1
        )[0]
        txt = random.choice(bucket).format(tag=tag)
        posts.append(txt)
    return posts

# -----------------------
# Gemini generation
# -----------------------
def generate_with_gemini(hashtag: str, n: int) -> List[str]:
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not set")

    model = genai.GenerativeModel("gemini-2.0-flash")
    tag = hashtag if hashtag.startswith("#") else f"#{hashtag}"

    prompt = f"""
You are generating short, natural social posts (Twitter/Instagram style) about the topic {tag}.
Rules:
- Return exactly {n} posts.
- One post per line.
- Each post under 120 characters.
- Use a mix of positive, neutral, and critical tones.
- Avoid hate speech or slurs.
- Do NOT include numbering or code blocks.
- Language: English.
Output format:
<post 1>
<post 2>
...
<post {n}>
"""
    tries = 2
    for i in range(tries):
        try:
            r = model.generate_content(prompt)
            text = (r.text or "").strip()
            lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
            if len(lines) < n:
                lines += make_fallback_posts(hashtag, n - len(lines))
            return lines[:n]
        except Exception as e:
            print(f"Gemini generation attempt {i+1} failed:", e)
            time.sleep(0.8)
            if i == tries - 1:
                raise

# -----------------------
# API: analyze
# -----------------------
@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json(silent=True) or {}
        hashtag = (data.get("hashtag") or "").strip()
        count = normalize_count(data.get("count") or DEFAULT_POSTS)

        if not hashtag:
            return jsonify({"error": "hashtag is required"}), 400

        posts: List[Dict] = []
        gemini_count = 0
        fallback_count = 0

        # Try Gemini first
        try:
            gemini_posts = generate_with_gemini(hashtag, count)
            for p in gemini_posts:
                posts.append({"text": p, "source": "gemini"})
            gemini_count = len(gemini_posts)
        except Exception:
            fb = make_fallback_posts(hashtag, count)
            for p in fb:
                posts.append({"text": p, "source": "fallback"})
            fallback_count = len(fb)

        # Sentiment analysis
        rows = []
        for p in posts:
            if sentiment_analyzer:
                res = sentiment_analyzer(p["text"])[0]
                parsed = parse_sentiment(res["label"], res["score"])
            else:
                parsed = {"sentiment": "NEUTRAL", "score": 0.5}
            rows.append({
                "text": p["text"],
                "source": p["source"],
                "sentiment": parsed["sentiment"],
                "score": parsed["score"],
            })

        agg = compute_aggregate(rows)

        return jsonify({
            "meta": {
                "hashtag": hashtag if hashtag.startswith("#") else f"#{hashtag}",
                "requested": count,
                "generated_by": {"gemini": gemini_count, "fallback": fallback_count},
                "model": {
                    "generation": "gemini-2.0-flash" if gemini_count > 0 else "fallback-templates",
                    "sentiment": SENTIMENT_MODEL
                }
            },
            "rows": rows,
            "aggregate": agg
        }), 200

    except Exception as e:
        print("API analyze error:", e)
        return jsonify({"error": str(e)}), 500

# -----------------------
# UI Route
# -----------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# -----------------------
# Entrypoint
# -----------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    app.run(host="0.0.0.0", port=port, debug=True)  # debug=True for detailed error logs
