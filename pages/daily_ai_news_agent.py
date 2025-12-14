import sys
import os
import json
import httpx
import torch
import hashlib
from datetime import datetime, timedelta
from transformers import pipeline
from openai import OpenAI
import streamlit as st
from pathlib import Path
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

# -----------------------------
# Ensure repo root is in sys.path
# -----------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
from utils import load_credentials, create_or_send_message


AUTO_RUN = os.getenv("AUTO_RUN", "false").lower() == "true"

# -----------------------------
# Environment & OpenAI client
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
if not OPENAI_API_KEY or not NEWS_API_KEY:
    raise RuntimeError("OPENAI_API_KEY or NEWS_API_KEY not set in env.")

client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# Sentiment analyzer (CPU-only)
# -----------------------------
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1
)
if hasattr(torch, "compile"):
    torch._dynamo.disable()

# -----------------------------
# Cache
# -----------------------------
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

def get_cache_key(articles):
    urls = [a.get("url", "") for a in articles]
    return hashlib.md5("".join(urls).encode()).hexdigest()

def save_cache(key, articles):
    with open(CACHE_DIR / f"{key}.json", "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

def is_cached(key):
    return (CACHE_DIR / f"{key}.json").exists()

def load_cache(key):
    with open(CACHE_DIR / f"{key}.json", "r", encoding="utf-8") as f:
        return json.load(f)

# -----------------------------
# AI Terms
# -----------------------------
AI_TERMS = [
    "AI", "artificial intelligence", "machine learning", "ChatGPT", "OpenAI",
    "large language model", "LLM", "deep learning", "neural network",
    "cloud", "HPC", "distributed inference"
]

# -----------------------------
# Fetch AI news
# -----------------------------
def get_ai_news_articles(tickers, max_articles=20):
    yesterday = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    url = "https://newsapi.org/v2/everything"
    query = "(" + " OR ".join(AI_TERMS) + ")"

    params = {
        "q": query,
        "from": yesterday,
        "sortBy": "publishedAt",
        "pageSize": max_articles,
        "apiKey": NEWS_API_KEY,
    }

    try:
        resp = httpx.get(url, params=params, timeout=30)
        resp.raise_for_status()
    except Exception as ex:
        st.error(f"Failed to fetch news: {ex}")
        return []

    articles = resp.json().get("articles", [])
    scored = []

    for a in articles:
        title = a.get("title", "")
        desc = a.get("description") or ""
        text = f"{title} {desc}"[:500].lower()
        relevance = sum(2 * text.count(term.lower()) for term in AI_TERMS)
        relevance += sum(5 for t in tickers if t.lower() in text)
        sentiment = sentiment_analyzer(text[:512])[0]
        scored.append({
            "title": title,
            "url": a.get("url"),
            "description": desc[:300],
            "relevance": relevance,
            "sentiment": sentiment
        })

    return sorted(scored, key=lambda x: x["relevance"], reverse=True)

# -----------------------------
# Format for GPT
# -----------------------------
def format_articles_for_gpt(articles, top_n=10):
    blocks = []
    for idx, a in enumerate(articles[:top_n], 1):
        blocks.append(
            f"Article {idx}:\n"
            f"Title: {a['title']}\n"
            f"Description: {a['description']}\n"
            f"URL: {a['url']}\n"
            f"Relevance Score: {a['relevance']}\n"
            f"Sentiment: {a['sentiment']['label']} (confidence {a['sentiment']['score']:.2f})"
        )
    return "\n\n".join(blocks)

# -----------------------------
# GPT analysis
# -----------------------------
def analyze_news_gpt(articles_text, tickers):
    system_prompt = f"""
You are a financial analyst specializing in AI/technology stocks.
Analyze these AI news articles for potential market impact.
Provide Buy/Hold/Watch signals for all tickers listed: {', '.join(tickers)}.
Write in a professional email format.
Prioritize the top 10 most relevant articles.
Use cautious language; all output is for human review only.
"""
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":f"Analyze these AI news articles:\n{articles_text}"}
        ],
        max_tokens=1000, temperature=0
    )
    analysis = resp.choices[0].message.content
    tokens_used = resp.usage.total_tokens
    cost = tokens_used / 1000 * 0.002
    return analysis, tokens_used, cost

# -----------------------------
# Gmail credentials (headless)
# -----------------------------
def load_credentials():
    token_file = "token.json"
    if os.path.exists(token_file):
        creds_data = json.load(open(token_file))
        creds = Credentials.from_authorized_user_info(creds_data, ['https://www.googleapis.com/auth/gmail.compose'])
    else:
        raise RuntimeError("token.json missing. Generate locally and set as secret for GitHub Actions.")

    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open(token_file, "w") as f:
            f.write(creds.to_json())
    if not creds.valid:
        raise RuntimeError("Invalid Gmail credentials. Please refresh token.json locally.")
    return creds

# -----------------------------
# Automatic run (GitHub Actions)
# -----------------------------
if AUTO_RUN:
    creds = load_credentials()
    recipients = [e.strip() for e in os.getenv("EMAIL_RECIPIENTS", "").split(",") if e.strip()]
    subject = f"[ADVISORY] AI Market Analysis - {datetime.utcnow().strftime('%B %d, %Y')}"
    TICKERS = ["AAPL","MSFT","GOOG","NVDA","TSLA","AMZN"]
    articles = get_ai_news_articles(TICKERS)
    analysis, tokens, cost = analyze_news_gpt(format_articles_for_gpt(articles, 10), TICKERS)
    for idx, recipient in enumerate(recipients, 1):
        result = create_or_send_message(creds, recipient, subject, analysis, advisor_id=f"advisor{idx}")
        print("Auto-run draft created:", result)
