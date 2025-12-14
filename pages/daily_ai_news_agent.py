
import sys
from pathlib import Path
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

# -----------------------------
# Ensure repo root is in sys.path
# -----------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]  # one level above pages/
sys.path.insert(0, str(ROOT_DIR))

# Now import utils safely
from utils import load_credentials, create_gmail_draft

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
sentiment_analyzer = pipeline("sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english", device=-1)
if hasattr(torch, "compile"):
    torch._dynamo.disable()

# -----------------------------
# Cache
# -----------------------------
CACHE_DIR = pathlib.Path("cache"); CACHE_DIR.mkdir(exist_ok=True)

def get_cache_key(articles):
    urls = [a.get("url","") for a in articles]
    return hashlib.md5("".join(urls).encode()).hexdigest()

def save_cache(key, articles):
    with open(CACHE_DIR/f"{key}.json","w",encoding="utf-8") as f:
        json.dump(articles,f,ensure_ascii=False,indent=2)

def is_cached(key): return (CACHE_DIR/f"{key}.json").exists()
def load_cache(key): return json.load(open(CACHE_DIR/f"{key}.json","r",encoding="utf-8"))

# -----------------------------
# Expanded AI + Infrastructure Terms
# -----------------------------
# -----------------------------
# AI + Software/ML Terms Only
# -----------------------------
AI_TERMS = [
    "AI", "artificial intelligence", "machine learning", "ChatGPT", "OpenAI",
    "large language model", "LLM", "deep learning", "neural network",
    "cloud", "HPC", "distributed inference"
]


def get_ai_news_articles(tickers, max_articles=20):
    yesterday = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    url = "https://newsapi.org/v2/everything"

    # Broad AI terms query without requiring tickers
    query = "(" + " OR ".join(AI_TERMS) + ")"

    params = {
        "q": query,
        "from": yesterday,
        "sortBy": "publishedAt",
        "pageSize": max_articles,
        "apiKey": NEWS_API_KEY,
        # Remove 'language':'en' to fetch all languages
    }

    resp = httpx.get(url, params=params, timeout=30)
    resp.raise_for_status()
    articles = resp.json().get("articles", [])
    scored = []

    for a in articles:
        title = a.get("title","")
        desc = a.get("description","") or ""
        text = f"{title} {desc}"[:500].lower()
        relevance = 0
        # weight AI terms only
        for term in AI_TERMS:
            relevance += 2 * text.count(term.lower())
        # optional: increase relevance if tickers appear
        for t in tickers:
            if t.lower() in text:
                relevance += 5
        sentiment = sentiment_analyzer(text[:512])[0]
        scored.append({"title":title,"url":a.get("url"),
                       "description":desc[:300],"relevance":relevance,
                       "sentiment":sentiment})

    # sort by relevance descending
    return sorted(scored, key=lambda x: x["relevance"], reverse=True)

def format_articles_for_gpt(articles,top_n=10):
    formatted_blocks=[]
    for idx,a in enumerate(articles[:top_n],start=1):
        block=(
            f"Article {idx}:\n"
            f"Title: {a['title']}\n"
            f"Description: {a['description']}\n"
            f"URL: {a['url']}\n"
            f"Relevance Score: {a['relevance']}\n"
            f"Sentiment: {a['sentiment']['label']} "
            f"(confidence {a['sentiment']['score']:.2f})"
        )
        formatted_blocks.append(block)
    return "\n\n".join(formatted_blocks)

def analyze_news_gpt(articles_text,tickers):
    system_prompt=f"""
You are a financial analyst specializing in AI/technology stocks.
Analyze these AI news articles for potential market impact.
Provide Buy/Hold/Watch signals for all tickers listed: {', '.join(tickers)}.
Write in a professional email format.
Prioritize the top 10 most relevant articles.
Use cautious language; all output is for human review only.
"""
    resp=client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"system","content":system_prompt},
                  {"role":"user","content":f"Analyze these AI news articles:\n{articles_text}"}],
        max_tokens=1000,temperature=0
    )
    analysis=resp.choices[0].message.content
    tokens_used=resp.usage.total_tokens
    cost=tokens_used/1000*0.002
    return analysis,tokens_used,cost

# -----------------------------
# UI
# -----------------------------
st.title("AI Market News Agent — Manual + Dashboard")

# --- Section 1: Manual Run ---
st.header("Manual Analysis & Draft Creation")
tickers_input=st.text_input("Tickers (comma-separated):","AAPL,MSFT,GOOG,NVDA,TSLA,AMZN")
TICKERS=[t.strip().upper() for t in tickers_input.split(",") if t.strip()]
email_input=st.text_input("Advisor email(s):","irinaswofford@gmail.com")

if st.button("Fetch & Analyze AI News"):
    articles=get_ai_news_articles(TICKERS)
    if not articles: st.error("No articles found.")
    else:
        st.text_area("Top Articles:",format_articles_for_gpt(articles,20),height=300)
        analysis,tokens,cost=analyze_news_gpt(format_articles_for_gpt(articles,10),TICKERS)
        st.text_area("GPT Draft Analysis:",analysis,height=400)
        st.info(f"Tokens used: {tokens} • Cost: ${cost:.4f}")
        approve=st.checkbox("I understand this is advisory only (required to create draft)")
        if approve and email_input:
            creds=load_credentials()
            recipients=[e.strip() for e in email_input.split(",") if e.strip()]
            subject=f"[ADVISORY] AI Market Analysis - {datetime.utcnow().strftime('%B %d, %Y')}"
            for idx,recipient in enumerate(recipients,1):
                draft=create_gmail_draft(creds,recipient,subject,analysis,advisor_id=f"advisor{idx}")
                if isinstance(draft,dict) and draft.get("id"):
                    st.success(f"Draft created for {recipient} (ID: {draft['id']})")
                else:
                    st.error(f"Failed to create draft for {recipient}: {draft}")

st.info("""
⚠️ Important:
- Manual runs only create drafts if you check the box above.
- Automatic job runs daily at 5 AM ET via GitHub Actions.
- That scheduled job always creates Gmail drafts and writes to the audit log.
- All drafts are advisory only and require human review before any action.
""")

# --- Section 2: Dashboard ---
st.header("Advisory Drafts Dashboard (Audit Log)")
AUDIT_LOG="draft_audit_log.json"
path=pathlib.Path(AUDIT_LOG)
if not path.exists():
    st.warning("No audit log found yet. Once the daily job runs, entries will appear here.")
else:
    entries=[]
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try: entries.append(json.loads(line))
                except: pass
    if not entries: st.info("Audit log present but empty.")
    else:
        for e in reversed(entries[-50:]):
            relevance_display = f"{e.get('relevance','?')}"
            sentiment_display = e.get('sentiment',{}).get('label','?')
            st.markdown(
                f"- **Advisor ID:** {e.get('advisor_id')} | **Recipient:** {e.get('recipient')}  \n"
                f"  **Subject:** {e.get('subject')} | **Time (UTC):** {e.get('timestamp')} | "
                f"**Draft ID:** {e.get('draft_id')} | **Relevance:** {relevance_display} | **Sentiment:** {sentiment_display}"
            )
        st.caption("Open Gmail to review drafts before sending. This dashboard is read-only.")
