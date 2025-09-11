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
from huggingface_hub import InferenceClient
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

@st.cache_resource(show_spinner=False)
def load_hf_llm():
    return InferenceClient(
        model="google/flan-t5-large",
        token=st.secrets["huggingface"]["api_key"]
    )


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

preferred_sources = list(SOURCE_FEEDS.keys())+list("Preferred sources")
preferred_sources1 = [
    'fierce pharma',
    'biopharma dive',
    'contract pharma',
    'fierce biotech',
    'pharmavoice',
    'biospace',
    'pharmaceutical technology',
    'endpoints news',
    'life science leader',
'crunchbase news',
    'endpoints news',
    'pharmavoice',
    'insights.siteline',
    'pharmaceutical technology',
    'life science leader',

    
]


# -------------------------------
# --- Helper Functions
# -------------------------------
def source_priority(source):
    source_lower = str(source).lower()
    return 0 if any(pref in source_lower for pref in preferred_sources1) else 1

def classify_with_embeddings(headline):
    text = re.sub(r'[^a-z\s]', '', str(headline).lower())
    headline_embedding = model.encode([text])
    similarities = cosine_similarity(headline_embedding, category_embeddings)[0]
    max_idx = np.argmax(similarities)
    max_sim = similarities[max_idx]
    return category_names[max_idx] if max_sim > 0.3 else "Other"



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
   
  
    import pytz
    articles = []
    today = datetime.now().date()

    # Setup datetime range for filtering based on timeline_choice
    if timeline_choice == "Today":
        start_datetime = datetime.combine(today, datetime.min.time())
        end_datetime = datetime.combine(today, datetime.max.time())
        date_range = (start_datetime, end_datetime)
    elif timeline_choice == "Yesterday":
        yesterday = today - timedelta(days=1)
        start_datetime = datetime.combine(yesterday, datetime.min.time())
        end_datetime = datetime.combine(yesterday, datetime.max.time())
        date_range = (start_datetime, end_datetime)
    elif timeline_choice == "Last 7 Days":
        start_datetime = datetime.combine(today - timedelta(days=6), datetime.min.time())
        end_datetime = datetime.combine(today, datetime.max.time())
        date_range = (start_datetime, end_datetime)
    elif timeline_choice == "Last 1 Month":
        start_datetime = datetime.combine(today - timedelta(days=29), datetime.min.time())
        end_datetime = datetime.combine(today, datetime.max.time())
        date_range = (start_datetime, end_datetime)
    elif timeline_choice == "Custom Range" and start_date and end_date:
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.max.time())
        date_range = (start_datetime, end_datetime)
    else:
        date_range = None  # No filtering

    try:
        feed = feedparser.parse(rss_url)

        for entry in feed.entries:
            
            # Extract published datetime (try published_parsed or updated_parsed)
            published_parsed = entry.get("published_parsed") or entry.get("updated_parsed")
            if published_parsed:
                published_dt = datetime(*published_parsed[:6])
            else:
                published_dt = None

            # Filter by timeline
            if date_range is not None:
                if not published_dt or not (date_range[0] <= published_dt <= date_range[1]):
                    continue

            headline = clean_html(entry.title)
            summary = clean_html(entry.get("summary", ""))

            content = f"{headline} {summary}".lower()

            if "content" in entry:
                try:
                    feed_content = " ".join([c.value for c in entry.content])
                    content += " " + clean_html(feed_content).lower()
                except Exception:
                    pass

            # Keyword filtering if keywords provided
            if keywords:
                kw_list = [kw.lower() for kw in keywords]

                def keyword_match(text):
                    if search_logic == "OR":
                        return any(kw in text for kw in kw_list)
                    elif search_logic == "AND":
                        return all(kw in text for kw in kw_list)
                    return False

                if not keyword_match(content):
                    # Fallback: load full article text and check keywords there
                    
                    loader = UnstructuredURLLoader(urls=[entry.link])
                    docs = loader.load()
                    full_text = " ".join(doc.page_content for doc in docs).lower()

                    if not keyword_match(full_text):
                        
                        continue  # skip article if no match

                

            articles.append({
                'Keyword': "Entered Keyword",
                'Headline': headline,
                'URL': entry.link,
                'Published on': published_dt.date() if published_dt else None,
                'Source': "Preferred sources",
                'HasExpandedKeyword': True
            })

    except Exception as e:
        logging.error(f"RSS fetch failed for {source}: {e}")

    return articles
def get_related_keywords(keyword, top_n=5):
    try:
        client = load_hf_llm()
        prompt = (
            f"You are an expert in biotech and pharma domains.\n"
            f"If '{keyword}' is not a company name, list {top_n} domain-specific keywords related to it.\n"
            f"If it's a company, provide the full name and known subsidiaries.\n"
            f"Respond ONLY with a comma-separated list."
        )

        # Use the conversational endpoint instead of text_generation
        response = client.conversational(
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_new_tokens=100,
            temperature=0.7,
        )

        text = response.strip()

        # Clean and extract comma-separated keywords
        keywords = re.split(r'[,\n]', text)
        keywords = [re.sub(r'^\d+\.?\s*', '', kw.strip()) for kw in keywords]
        keywords = [kw for kw in keywords if kw and keyword.lower() not in kw.lower()]
        return list(dict.fromkeys(keywords))[:top_n]

    except Exception as e:
        logging.error(f"Hugging Face LLM keyword generation failed for {keyword}: {e}")
        return []

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
    max_articles = st.number_input("Max articles to display  (up to 5000)", min_value=10, max_value=5000, value=100, step=10)
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
                    max_articles=1000,
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
                        max_articles=1000,
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
    df['Source']=np.where(df['Source'].isin(preferred_sources1),"Preferred sources",df['Source'])

   
    df = df.sort_values(by=['priority','HasExpandedKeyword', 'Published on'], ascending=[True,False, False])

    # Drop helper cols and dedupe
    df = df.drop(columns=['priority'], errors='ignore')
    
    

    df = df.drop_duplicates(subset=['Headline'])
    df=df.head(max_articles)

    st.session_state['articles_df'] = df
    if "Preferred sources" in list(df['Source']):
        filtered_df = df[df['Source'] == 'Preferred sources']
        st.session_state['filtered_df'] = filtered_df
    else:
        
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
        default_source = "Preferred sources" if "Preferred sources" in available_sources else "All"
        source_filter = st.selectbox("🔗 Source", options=["All"] + available_sources, index=(["All"] + available_sources).index(default_source))


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
           
