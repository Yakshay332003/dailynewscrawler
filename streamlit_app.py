import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import feedparser
import requests
import re
from urllib.parse import quote
import logging
import numpy as np
from langchain.document_loaders import UnstructuredURLLoader
from langchain.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time
import torch
from langchain.tools import Tool
from transformers import pipeline
from bs4 import BeautifulSoup
import httpx
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# -------------------------------
# --- Constants and Model Loading
# -------------------------------
CATEGORIES = {
    "Drug Pipeline": ['fda', 'trial', 'phase', 'approval', 'clinical', 'study', 'enrollment', 'efficacy'],
    "Investments": ['funding', 'investment', 'raises', 'venture', 'capital', 'backing', 'grant'],
    "M&A": ['acquisition', 'merger', 'buyout', 'acquire', 'deal', 'takeover'],
    "Partnerships": ['partnership', 'collaboration', 'agreement', 'alliance', 'teaming up'],
    "Financials": ['earnings', 'revenue', 'profit', 'loss', 'q1', 'q2', 'quarter', 'forecast', 'financial'],
}

@st.cache_resource(show_spinner=False)
def load_sentence_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_sentence_model()
category_texts = {cat: " ".join(words) for cat, words in CATEGORIES.items()}
category_embeddings = model.encode(list(category_texts.values()))
category_names = list(category_texts.keys())

@st.cache_resource(show_spinner=False)
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# -------------------------------


summarizer_pipeline = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
llm = HuggingFacePipeline(pipeline=summarizer_pipeline)
def read_url(url: str) -> str:
    try:
        response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = "\n".join(p.get_text() for p in paragraphs)
        return text[:4000]  # Limit for context window
    except Exception as e:
        return f"Error fetching URL: {e}"

url_reader_tool = Tool(
    name="URL Reader",
    func=read_url,
    description="Fetches and returns the readable content of a news article from a URL"
)
def resolve_google_news_url(google_news_url):
    try:
        # Allow redirects to follow
        response = requests.get(google_news_url, timeout=10)
        return response.url  # Final redirected URL
    except Exception as e:
        logging.error(f"Error resolving original URL: {e}")
        return google_news_url  # fallback to original

# --- Helper Functions
# -------------------------------
def summarize_url(url):
    content = read_url(url)
    if "Error" in content:
        return content
    summary = summarizer_pipeline(content[:1024])[0]['summary_text']
    return summary
def classify_with_embeddings(headline):
    text = re.sub(r'[^a-z\s]', '', headline.lower())
    headline_embedding = model.encode([text])
    similarities = cosine_similarity(headline_embedding, category_embeddings)[0]
    max_idx = np.argmax(similarities)
    max_sim = similarities[max_idx]
    return category_names[max_idx] if max_sim > 0.3 else "Other"

def fetch_latest_headlines_rss(keyword, max_results, timeline_choice="All", start_date=None, end_date=None):
    articles = []
    today = datetime.now().date()

    # Determine the date range based on timeline_choice
    if timeline_choice == "Today":
        date_range = [today]
    elif timeline_choice == "Yesterday":
        date_range = [today - timedelta(days=1)]
    elif timeline_choice == "Last 7 Days":
        date_range = [today - timedelta(days=i) for i in range(7)]
    elif timeline_choice == "Last 1 Month":
        date_range = [today - timedelta(days=i) for i in range(30)]
    elif timeline_choice == "Custom Range" and start_date and end_date:
        date_range = pd.date_range(start=start_date, end=end_date).to_pydatetime().tolist()
        date_range = [d.date() for d in date_range]
    else:
        date_range = [None]  # No specific filter

    for date_val in date_range:
        
        query = f'{keyword} after:{date_val - timedelta(days=1)} before:{date_val + timedelta(days=1)}'

        rss_url = f"https://news.google.com/rss/search?q={quote(query)}&hl=en-IN&gl=IN&ceid=IN:en"

        try:
            response = requests.get(rss_url, timeout=10)
            response.raise_for_status()
            feed = feedparser.parse(response.content)

            for entry in feed.entries:
                published_at = datetime(*entry.published_parsed[:6]) if entry.get("published_parsed") else None
                source = entry.get('source', {}).get('title', 'Unknown')
                source = str(source).lower().replace(".com", "")

                articles.append({
                    'Keyword': keyword,
                    'Headline': entry.title,
                    'URL': resolve_google_news_url(entry.link),
                    'Published on': published_at,
                    'Source': source
                })

            if len(articles) >= max_results:
                break

        except requests.RequestException as e:
            logging.error(f"Request failed for {keyword}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error for {keyword}: {e}")
        
        

        if timeline_choice != "All":
            today = datetime.now().date()
            if timeline_choice == "Today":
                articles = [a for a in articles if a['Published on'] and a['Published on'].date() == today]
            elif timeline_choice == "Yesterday":
                yday = today - timedelta(days=1)
                articles = [a for a in articles if a['Published on'] and a['Published on'].date() == yday]
            elif timeline_choice == "Last 7 Days":
                week_ago = today - timedelta(days=7)
                articles = [a for a in articles if a['Published on'] and a['Published on'].date() >= week_ago]
            elif timeline_choice == "Last 1 Month":
                month_ago = today - timedelta(days=30)
                articles = [a for a in articles if a['Published on'] and a['Published on'].date() >= month_ago]
            elif timeline_choice == "Custom Range" and start_date and end_date:
                articles = [a for a in articles if a['Published on'] and start_date <= a['Published on'].date() <= end_date]

    return articles[:max_results]

def get_final_article_url_selenium(url):
    try:
        response = httpx.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        meta = soup.find("meta", attrs={"http-equiv": "refresh"})
        if meta and "content" in meta.attrs:
            content = meta["content"]
            if "url=" in content.lower():
                real_url = content.split("URL=")[-1].strip()
                return real_url
        return url
    except Exception as e:
        print("Failed to resolve real URL:", e)
        return url

def extract_article_text(url):
    try:
        loader = UnstructuredURLLoader(urls=[url])
        docs = loader.load()
        return docs[0].page_content if docs else ""
    except Exception as e:
        return f"Error extracting content: {e}"

# -------------------------------
# --- Streamlit App
# -------------------------------
st.set_page_config(page_title="📰 Keyword News Explorer with Summarization", layout="wide")
st.title("📰 Keyword News Explorer with Summarization")

# Input Section
with st.form("fetch_form"):
    keywords_input = st.text_area("🔍 Enter keywords (comma-separated)", placeholder="e.g., Pfizer, biotech, gene therapy")
    max_articles = st.number_input("Max articles per keyword (up to 500)", min_value=10, max_value=1000, value=100, step=10)
    timeline_choice = st.selectbox("📆 Fetch Timeline", [ "Today", "Yesterday", "Last 7 Days", "Last 1 Month", "Custom Range"])
    start_date = end_date = None
    if timeline_choice == "Custom Range":
        start_date = st.date_input("From Date", value=datetime.now().date() - timedelta(days=7))
        end_date = st.date_input("To Date", value=datetime.now().date())
    submitted = st.form_submit_button("Fetch News")

# Fetch Articles
if submitted:
    if not keywords_input.strip():
        st.warning("Please enter at least one keyword.")
        st.stop()

    keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]
    all_articles = []

    with st.spinner("🔎 Fetching articles..."):
        for keyword in keywords:
            articles = fetch_latest_headlines_rss(
                keyword,
                max_results=max_articles,
                timeline_choice=timeline_choice,
                start_date=start_date,
                end_date=end_date
            )
            all_articles.extend(articles)

    if not all_articles:
        st.error("⚠️ No news found for the given keywords and timeline.")
        st.stop()

    df = pd.DataFrame(all_articles)
    df['Published on'] = pd.to_datetime(df['Published on'], errors='coerce')
    df['Category'] = df['Headline'].apply(classify_with_embeddings)
    df.sort_values(by="Published on", ascending=False, inplace=True)

    st.session_state['articles_df'] = df
    st.session_state['filtered_df'] = df

# Display Section
if 'articles_df' in st.session_state:
    df = st.session_state['articles_df'].copy()
    st.markdown("---")
    st.subheader("🧰 Filters")

    available_keywords = sorted(df['Keyword'].dropna().unique())
    available_sources = sorted(df['Source'].dropna().unique())
    available_categories = sorted(df['Category'].dropna().unique())

    col1, col2 = st.columns(2)
    with col1:
        keyword_filter = st.selectbox("🔑 Keyword", options=["All"] + available_keywords)
    with col2:
        source_filter = st.selectbox("🔗 Source", options=["All"] + available_sources)

    col3, _ = st.columns(2)
    with col3:
        category_filter = st.selectbox("🏷️ Category", options=["All"] + available_categories)

    if st.button("Apply Filters"):
        filtered_df = df.copy()

        if keyword_filter != "All":
            filtered_df = filtered_df[filtered_df['Keyword'] == keyword_filter]

        if source_filter != "All":
            filtered_df = filtered_df[filtered_df['Source'] == source_filter]

        if category_filter != "All":
            filtered_df = filtered_df[filtered_df['Category'] == category_filter]

        filtered_df.sort_values(by="Published on", ascending=False, inplace=True)
        st.session_state['filtered_df'] = filtered_df

    filtered_df = st.session_state.get('filtered_df', df)
    st.markdown(f"### 📄 Showing {len(filtered_df)} articles")

    def summarize_article(idx, url):
        with st.spinner("Generating summary..."):
            try:
                article_text= summarize_url(url)
                if len(article_text) > 1000:
                    article_text = article_text[:1000]
                if len(article_text.strip()) < 100:
                    st.session_state[f"summary_{idx}"] = url
                else:
                    summary = summarizer(article_text, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
                    st.session_state[f"summary_{idx}"] = summary
            except Exception as e:
                st.session_state[f"summary_{idx}"] = f"Failed to summarize: {e}"

    for idx, row in filtered_df.iterrows():
        with st.expander(f"🔹 {row['Headline']}", expanded=False):
            st.markdown(f"**Keyword:** {row['Keyword']}")
            st.markdown(f"**Published on:** {row['Published on'].strftime('%Y-%m-%d %H:%M') if pd.notnull(row['Published on']) else 'N/A'}")
            st.markdown(f"**Source:** {row['Source']}")
            st.markdown(f"**Category:** {row['Category']}")
            st.markdown(f"[🔗 Read Full Article]({row['URL']})")

            summary_key = f"summary_{idx}"
            if st.button("📝 Summarize Article", key=f"summarize_btn_{idx}"):
                summarize_article(idx, row['URL'])

            if summary_key in st.session_state:
                st.markdown("**Summary:**")
                st.write(st.session_state[summary_key])
