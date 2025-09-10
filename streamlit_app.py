import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import feedparser
import requests
import re
from urllib.parse import quote
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time
import torch
from bs4 import BeautifulSoup
import urllib3
import google.generativeai as genai
import os
from huggingface_hub import InferenceClient
from langchain.document_loaders import UnstructuredURLLoader  # used in extract_article_text
import re
from html import unescape
def clean_html(text):
    """Simple HTML cleaner (you may already have this)."""
    return unescape(BeautifulSoup(text, "html.parser").get_text())

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

def load_gemini_model():
    # expects st.secrets["gemini"]["api_key"] to be set
    genai.configure(api_key=st.secrets["gemini"]["api_key"])
    model = genai.GenerativeModel("gemini-2.5-pro")  # or "gemini-1.5-pro"
    return model

@st.cache_resource(show_spinner=False)
def load_sentence_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_sentence_model()

category_texts = {cat: " ".join(words) for cat, words in CATEGORIES.items()}
category_embeddings = model.encode(list(category_texts.values()))
category_names = list(category_texts.keys())

# -------------------------------
# Preferred sources (manual RSS feeds)
# -------------------------------
SOURCE_FEEDS = {
    "fierce pharma": "https://www.fiercepharma.com/rss/xml",
    "biopharma dive": "https://www.biopharmadive.com/rss/",
    "contract pharma": "https://www.contractpharma.com/rss/",
    "fierce biotech": "https://www.fiercebiotech.com/rss/xml",
    "pharmavoice": "https://www.pharmavoice.com/rss",
    "biospace": "https://www.biospace.com/rss/",
    "pharmaceutical technology": "https://www.pharmaceutical-technology.com/feed/",
    "endpoints news": "https://endpts.com/feed/",
    "life science leader": "https://www.lifescienceleader.com/rss/allcontent.aspx",
    "crunchbase news": "https://news.crunchbase.com/feed/"
}

preferred_sources = list(SOURCE_FEEDS.keys())

# -------------------------------
# --- Helper Functions
# -------------------------------
def source_priority(source):
    source_lower = str(source).lower()
    return 0 if any(pref in source_lower for pref in preferred_sources) else 1

def classify_with_embeddings(headline):
    text = re.sub(r'[^a-z\s]', '', str(headline).lower())
    headline_embedding = model.encode([text])
    similarities = cosine_similarity(headline_embedding, category_embeddings)[0]
    max_idx = np.argmax(similarities)
    max_sim = similarities[max_idx]
    return category_names[max_idx] if max_sim > 0.3 else "Other"

def get_related_keywords(keyword, top_n=5):
    try:
        gem = load_gemini_model()
        prompt = (
            f"If '{keyword}' is not a company name, list {top_n} distinct, domain-specific keywords related to '{keyword}'. "
            f"If '{keyword}' is a company name, give the full company name and list any known subsidiaries. "
            f"Respond with a comma-separated list only."
        )
        response = gem.generate_content(prompt)
        text = response.text.strip()
        # parse
        keywords = re.split(r'[,\n]', text)
        keywords = [re.sub(r'^\d+\.?\s*', '', kw.strip()) for kw in keywords]
        keywords = [kw for kw in keywords if kw and keyword.lower() not in kw.lower()]
        return list(dict.fromkeys(keywords))[:top_n]
    except Exception as e:
        logging.error(f"Gemini keyword generation failed for {keyword}: {e}")
        return []

def fetch_latest_headlines_rss(keyword, max_articles=100, timeline_choice="All", start_date=None, end_date=None):
    """
    Google News RSS search (keeps your original implementation but made robust to timeline filtering).
    `keyword` is a string (can be combined quotes for AND).
    """
    articles = []
    today = datetime.now().date()

    # Determine date_range (for compatibility we support 'Today','Yesterday','Last 7 Days','Last 1 Month','Custom Range')
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
        date_range = [None]

    for date_val in date_range:
        if date_val:
            q = f'{keyword} after:{date_val - timedelta(days=1)} before:{date_val + timedelta(days=1)}'
        else:
            q = keyword

        rss_url = f"https://news.google.com/rss/search?q={quote(q)}&hl=en-IN&gl=IN&ceid=IN:en"
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(rss_url, timeout=10, headers=headers)
            response.raise_for_status()
            feed = feedparser.parse(response.content)

            for entry in feed.entries:
                published_at = None
                if entry.get("published_parsed"):
                    published_at = datetime(*entry.published_parsed[:6])
                source = entry.get('source', {}).get('title', 'Unknown')
                source = str(source).lower().replace(".com", "")
                articles.append({
                    'Keyword': keyword,
                    'Headline': clean_html(entry.title),
                    'URL': entry.link,
                    'Published on': published_at,
                    'Source': source
                })
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
                

            # break early if we've collected enough (entries across dates will be appended)
            if len(articles) >= max_articles:
                break

        except requests.RequestException as e:
            logging.error(f"Request failed for {keyword}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error for {keyword}: {e}")

    return articles



def fetch_direct_rss(source, rss_url, max_articles=100, keywords=None,
                     timeline_choice="All", start_date=None, end_date=None, search_logic="OR"):
    articles = []
    today = datetime.now().date()

    # Determine date_range for timeline filtering
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
    elif timeline_choice == "All":
        date_range = None  # No date filtering
    else:
        # Default: no filtering if timeline_choice unrecognized
        date_range = None

    try:
        feed = feedparser.parse(rss_url)

        for entry in feed.entries:
            if len(articles) >= max_articles:
                break

            # Extract published date
            published_at = None
            if entry.get("published_parsed"):
                published_at = datetime(*entry.published_parsed[:6]).date()

            # Skip if date filtering applied and published_at not in range
            if date_range is not None:
                if not published_at or published_at not in date_range:
                    continue

            headline = clean_html(entry.title)
            summary = clean_html(entry.get("summary", ""))

            # Combine headline and summary content
            content = f"{headline} {summary}".lower()

            # Add content from 'content' field if present
            if "content" in entry:
                try:
                    feed_content = " ".join([c.value for c in entry.content])
                    content += " " + clean_html(feed_content).lower()
                except Exception:
                    pass

            # Keyword filtering
            if keywords:
                kw_list = [kw.lower() for kw in keywords]

                def keyword_match(text):
                    if search_logic == "OR":
                        return any(kw in text for kw in kw_list)
                    elif search_logic == "AND":
                        return all(kw in text for kw in kw_list)
                    return False

                if not keyword_match(content):
                    try:
                        loader = UnstructuredURLLoader(urls=[entry.link])
                        docs = loader.load()
                        full_text = " ".join(doc.page_content for doc in docs).lower()

                        if not keyword_match(full_text):
                            continue  # no match, skip article

                    except Exception as e:
                        logging.warning(f"UnstructuredURLLoader failed for {entry.link}: {e}")
                        continue

            # Add relevant article
            articles.append({
                'Keyword': " ".join(keywords) if keywords else source,
                'Headline': headline,
                'URL': entry.link,
                'Published on': published_at,
                'Source': "Preferred sources",
                'HasExpandedKeyword': True
            })

    except Exception as e:
        logging.error(f"RSS fetch failed for {source}: {e}")

    return articles

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
st.set_page_config(page_title="📰 Hunt News", layout="wide")
st.title("📰 Hunt News by Keyword ")

# Input Section
with st.form("fetch_form"):
    keywords_input = st.text_area("🔍 Enter keywords (comma-separated)", placeholder="e.g., Pfizer, biotech, gene therapy")
    max_articles = st.number_input("Max articles to display  (up to 1000)", min_value=10, max_value=1000, value=100, step=10)
    timeline_choice = st.selectbox("📆 Fetch Timeline", [ "Today", "Yesterday", "Last 7 Days", "Last 1 Month", "Custom Range", "All"])
    search_mode = st.radio("🔎 Search Mode", ["Individual keywords (OR)", "All keywords together (AND)"], index=0)

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
        # --- Step 1: Fetch directly from preferred sources first
        # Use keyword filters on direct RSS according to search_mode and expansions
        if search_mode == "All keywords together (AND)":
            # Direct RSS: pass whole keywords list with AND logic
            for src, rss_url in SOURCE_FEEDS.items():
                direct = fetch_direct_rss(
                    source=src,
                    rss_url=rss_url,
                    max_articles=max_articles,
                    keywords=keywords,
                    timeline_choice=timeline_choice,
                    start_date=start_date,
                    end_date=end_date,
                    search_logic="AND"
                )
                # direct already has HasExpandedKeyword set
                all_articles.extend(direct)

            combined_query = " ".join([f'"{kw}"' for kw in keywords])
            g_articles = fetch_latest_headlines_rss(
                combined_query,
               1000,
                timeline_choice=timeline_choice,
                start_date=start_date,
                end_date=end_date
            )

            for a in g_articles:
                headline_text = str(a["Headline"]).lower()
                has_all_keywords = all(kw.lower() in headline_text for kw in keywords)
                has_expanded = has_all_keywords  # since this is AND mode
                a["HasExpandedKeyword"] = has_expanded
                a["Keyword"] =  " ".join(keywords) 
                all_articles.append(a)

        else:  # Individual keywords (OR)
            # For each keyword get related keywords and use OR logic for both direct RSS and Google News
            for keyword in keywords:
                related_kws = get_related_keywords(keyword, top_n=5)
                related_kws = list(dict.fromkeys(related_kws))  # preserve uniqueness
                st.write(f"Related keywords for **{keyword}**: {related_kws}") 

                expanded_keywords = related_kws + [keyword]

                # Direct RSS: for each source fetch using expanded_keywords with OR logic (will include if any match)
                for src, rss_url in SOURCE_FEEDS.items():
                    direct = fetch_direct_rss(
                        source=src,
                        rss_url=rss_url,
                        max_articles=max_articles,
                        keywords=expanded_keywords,
                        timeline_choice=timeline_choice,
                        start_date=start_date,
                        end_date=end_date,
                        search_logic="OR"
                    )
                    # direct includes HasExpandedKeyword which indicates whether any expanded kw matched
                    # tag the record's Keyword to show which base keyword triggered this (use keyword)
                    
                    all_articles.extend(direct)

                # Google News fallback: call for each expanded kw (OR-mode behavior you had before)
                for kw in expanded_keywords:
                    g_articles = fetch_latest_headlines_rss(
                        kw,
                        1000,
                        timeline_choice=timeline_choice,
                        start_date=start_date,
                        end_date=end_date
                    )
                    for a in g_articles:
                        headline_text = str(a["Headline"]).lower()
                        a["HasExpandedKeyword"] = any(ek.lower() in headline_text for ek in expanded_keywords)
                        a["Keyword"] = kw 
                        all_articles.append(a)

    if not all_articles:
        st.error("⚠️ No news found for the given keywords and timeline.")
        st.stop()

    # Final DataFrame operations
    df = pd.DataFrame(all_articles)

    # Ensure Published on is datetime
    df['Published on'] = pd.to_datetime(df['Published on'], errors='coerce')
    

  

    # Category classification and priority
    df['Category'] = df['Headline'].apply(classify_with_embeddings)
    df['priority'] = df['Source'].apply(source_priority)

    # Sort: prefer HasExpandedKeyword, then preferred source priority (0 first), then newest
    df = df.sort_values(by=['priority','HasExpandedKeyword', 'Published on'], ascending=[True,False, False])

    # Drop helper cols and dedupe
    df = df.drop(columns=['priority'], errors='ignore')
    df['Headline'] = df['Headline'].astype(str).apply(lambda x: x.split("-")[0])

    df = df.drop_duplicates(subset=['Headline'])

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

        st.session_state['filtered_df'] = filtered_df

    filtered_df = st.session_state.get('filtered_df', df)
    st.markdown(f"### 📄 Showing {len(filtered_df)} articles")

    for idx, row in filtered_df.iterrows():
        with st.expander(f"🔹 {row['Headline']}", expanded=False):
            st.markdown(f"**Keyword:** {row['Keyword']}")
            st.markdown(f"**Published on:** {row['Published on'].strftime('%Y-%m-%d %H:%M') if pd.notnull(row['Published on']) else 'N/A'}")
            st.markdown(f"**Source:** {row['Source']}")
            st.markdown(f"**Category:** {row['Category']}")
            st.markdown(f"[🔗 Read Full Article]({row['URL']})")
           
