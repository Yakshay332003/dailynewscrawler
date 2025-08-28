import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import feedparser
import requests
import re
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import torch
from transformers import pipeline
from bs4 import BeautifulSoup

# Disable SSL warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- Constants and Model Loading ---
CATEGORIES = {
    "Drug Pipeline": ['fda', 'trial', 'phase', 'approval', 'clinical', 'study', 'enrollment', 'efficacy'],
    "Investments": ['funding', 'investment', 'raises', 'venture', 'capital', 'backing', 'grant'],
    "M&A": ['acquisition', 'merger', 'buyout', 'acquire', 'deal', 'takeover'],
    "Partnerships": ['partnership', 'collaboration', 'agreement', 'alliance', 'teaming up'],
    "Financials": ['earnings', 'revenue', 'profit', 'loss', 'q1', 'q2', 'quarter', 'forecast', 'financial'],
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
category_texts = {cat: " ".join(words) for cat, words in CATEGORIES.items()}
category_embeddings = model.encode(list(category_texts.values()))
category_names = list(category_texts.keys())

# --- Helper functions ---

@st.cache_resource(show_spinner=False)
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

def classify_with_embeddings(headline):
    text = re.sub(r'[^a-z\s]', '', headline.lower())
    headline_embedding = model.encode([text])
    similarities = cosine_similarity(headline_embedding, category_embeddings)[0]
    max_idx = np.argmax(similarities)
    max_sim = similarities[max_idx]
    return category_names[max_idx] if max_sim > 0.3 else "Other"

def fetch_latest_headlines_rss(keyword):
    # Note: Google News RSS doesn't officially support pagination,
    # so this is a simple fetch - to get more articles, repeat with date filters or multiple keywords.
    rss_url = f"https://news.google.com/rss/search?q={keyword}"
    try:
        response = requests.get(rss_url, verify=False, timeout=10)
        response.raise_for_status()
        feed = feedparser.parse(response.content)
    except Exception:
        return []

    articles = []
    for entry in feed.entries:  # limit number of articles here
        try:
            published_at = datetime(*entry.published_parsed[:6])
        except Exception:
            published_at = None
        source = entry.get('source', {}).get('title', 'Unknown')
        source = str(source).lower().replace(".com", "")
        articles.append({
            'Keyword': keyword,
            'Headline': entry.title,
            'URL': entry.link,
            'Published on': published_at,
            'Source': source
        })
    return articles

def filter_by_timeline(df, timeline_choice, start_date=None, end_date=None):
    df = df.copy()
    df['PublishedDate'] = df['Published on'].dt.date
    today = datetime.now().date()

    if timeline_choice == "Today":
        return df[df['PublishedDate'] == today]
    elif timeline_choice == "Yesterday":
        return df[df['PublishedDate'] == today - timedelta(days=1)]
    elif timeline_choice == "Last 7 Days":
        return df[df['PublishedDate'] >= today - timedelta(days=7)]
    elif timeline_choice == "Last 1 Month":
        return df[df['PublishedDate'] >= today - timedelta(days=30)]
    elif timeline_choice == "Custom Range" and start_date and end_date:
        return df[(df['PublishedDate'] >= start_date) & (df['PublishedDate'] <= end_date)]
    return df

def get_final_article_url_selenium(url):
    # This is a placeholder; you can improve by adding real Selenium code if needed.
    try:
        response = requests.head(url, allow_redirects=True, timeout=10)
        return response.url
    except Exception:
        return url

def extract_article_text(url):
    try:
        resp = requests.get(url, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = "\n".join([p.get_text() for p in paragraphs if len(p.get_text()) > 20])
        return text
    except Exception:
        return ""

# --- Streamlit UI ---

st.set_page_config(page_title="📰 Keyword News Explorer with Summarization", layout="wide")
st.title("📰 Keyword News Explorer with Summarization")

# Input section
with st.form("fetch_form"):
    keywords_input = st.text_area("🔍 Enter keywords (comma-separated)", placeholder="e.g., Pfizer, biotech, gene therapy")
    timeline_choice = st.selectbox("📆 Timeline", ["All", "Today", "Yesterday", "Last 7 Days", "Last 1 Month", "Custom Range"])
    start_date = end_date = None
    if timeline_choice == "Custom Range":
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("From Date")
        with col2:
            end_date = st.date_input("To Date")
        if start_date > end_date:
            st.warning("⚠️ 'From' date cannot be after 'To' date.")
            st.stop()
    max_articles = st.number_input("Max articles per keyword (up to 500)", min_value=10, max_value=500, value=100, step=10)
    submitted = st.form_submit_button("Fetch News")

if submitted:
    if not keywords_input.strip():
        st.warning("Please enter at least one keyword.")
        st.stop()

    keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]
    all_articles = []

    with st.spinner("🔎 Fetching articles..."):
        for keyword in keywords:
            # fetch more articles if max_articles is large
            articles = fetch_latest_headlines_rss(keyword, max_results=max_articles)
            all_articles.extend(articles)

    if not all_articles:
        st.error("⚠️ No news found for the given keywords.")
        st.stop()

    df = pd.DataFrame(all_articles)
    df['Published on'] = pd.to_datetime(df['Published on'], errors='coerce')
    df['Category'] = df['Headline'].apply(classify_with_embeddings)
    st.session_state['articles_df'] = df
    st.session_state['timeline_choice'] = timeline_choice
    st.session_state['start_date'] = start_date
    st.session_state['end_date'] = end_date
    st.session_state['filtered_df'] = None  # reset filters on new fetch

if 'articles_df' in st.session_state:
    df = st.session_state['articles_df'].copy()
    timeline_choice = st.session_state.get('timeline_choice', "All")
    start_date = st.session_state.get('start_date', None)
    end_date = st.session_state.get('end_date', None)

    st.markdown("---")
    st.subheader("🧰 Filters")

    available_keywords = ["All"] + sorted(df['Keyword'].unique())
    available_sources = ["All"] + sorted(df['Source'].dropna().unique())
    available_categories = ["All"] + sorted(df['Category'].dropna().unique())

    col1, col2, col3 = st.columns(3)
    with col1:
        keyword_filter = st.selectbox("🔑 Keyword", available_keywords)
    with col2:
        source_filter = st.selectbox("🔗 Source", available_sources)
    with col3:
        category_filter = st.selectbox("🏷️ Category", available_categories)

    if st.button("Apply Filters"):
        filtered_df = filter_by_timeline(df, timeline_choice, start_date, end_date)

        if keyword_filter != "All":
            filtered_df = filtered_df[filtered_df['Keyword'] == keyword_filter]
        if source_filter != "All":
            filtered_df = filtered_df[filtered_df['Source'].str.lower() == source_filter.lower()]
        if category_filter != "All":
            filtered_df = filtered_df[filtered_df['Category'] == category_filter]

        if filtered_df.empty:
            st.warning("⚠️ No articles match the selected filters.")
        st.session_state['filtered_df'] = filtered_df

    filtered_df = st.session_state.get('filtered_df', df)

    st.markdown(f"### 📄 Showing {len(filtered_df)} articles")

    # Summarization callback function
    def summarize_article(idx, url):
        with st.spinner("Generating summary..."):
            try:
                final_url = get_final_article_url_selenium(url)
                article_text = extract_article_text(final_url)
                if len(article_text) > 1000:
                    article_text = article_text[:1000]

                summary = summarizer(article_text, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
                st.session_state[f"summary_{idx}"] = summary
            except Exception as e:
                st.error(f"Failed to summarize: {e}")

    for idx, row in filtered_df.iterrows():
        with st.expander(f"🔹 {row['Headline']}", expanded=False):
            st.markdown(f"**Keyword:** {row['Keyword']}")
            st.markdown(f"**Published on:** {row['Published on'].strftime('%Y-%m-%d %H:%M') if pd.notnull(row['Published on']) else 'N/A'}")
            st.markdown(f"**Source:** {row['Source']}")
            st.markdown(f"**Category:** {row['Category']}")
            st.markdown(f"[🔗 Read Full Article]({row['URL']})")

            summary_key = f"summary_{idx}"

            # Button uses on_click to update state
            st.button("📝 Summarize Article", key=summary_key, on_click=summarize_article, args=(idx, row['URL']))

            if summary_key in st.session_state:
                st.markdown("**Summary:**")
                st.write(st.session_state[summary_key])
