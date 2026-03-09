import streamlit as st
import plotly.express as px
import time
import scraper
import analyzer
from collections import Counter
import re

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Reddit Radar | AI Brand Intelligence", page_icon="📡", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    h1 { color: #FF4500; font-family: 'Helvetica Neue', sans-serif; }
    .stButton>button { background-color: #FF4500; color: white; border-radius: 10px; width: 100%; }
    .stProgress > div > div > div > div { background-color: #00CC96; }
    
    /* Insights Box */
    .insight-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #262730;
        border-left: 5px solid #FF4500;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- HELPER: KEYWORD EXTRACTION ---
def extract_top_keywords(text_list):
    """Finds the most common meaningful words in a list of comments."""
    all_text = " ".join(text_list).lower()
    # Remove special chars
    all_text = re.sub(r'[^\w\s]', '', all_text)
    words = all_text.split()
    
    # Standard boring words to ignore
    stopwords = set(['the', 'is', 'and', 'to', 'a', 'of', 'in', 'it', 'for', 'that', 'on', 'with', 'was', 'as', 'are', 'but', 'this', 'have', 'be', 'my', 'not', 'just', 'so', 'iphone', 'phone', 'product', 'app', 'reddit', 'com', 'http', 'https', 'www'])
    
    filtered_words = [w for w in words if w not in stopwords and len(w) > 3]
    count = Counter(filtered_words)
    return count.most_common(5) # Return top 5 keywords

# --- HEADER ---
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/43/Reddit.svg", width=70)
with col2:
    st.title("Reddit Radar AI")
    st.caption("🚀 Advanced Brand Sentiment & Market Intelligence System")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
    mode = st.radio("Choose Analysis Mode:", ["🔍 Search Brand", "🔗 Analyze Link (Reddit/Insta)"])
    st.info("Tip: 'Analyze Link' dives deep into comments of a specific post.")

# --- INPUT SECTION ---
st.markdown("---")
df = None

if mode == "🔍 Search Brand":
    topic = st.text_input("Enter Brand Name:", placeholder="e.g., iPhone 16, GTA VI")
    if st.button("🚀 SCAN REDDIT"):
        with st.status("🤖 Scanning Ecosystem...", expanded=True) as status:
            df = scraper.fetch_reddit_search(topic)
            if df is not None and not df.empty:
                st.write("✅ Search Complete.")
                st.write("🧠 Running Sentiment AI...")
                df = analyzer.run_analysis(df)
                status.update(label="✅ Analysis Ready!", state="complete", expanded=False)
            else:
                status.update(label="❌ No data found.", state="error")

elif mode == "🔗 Analyze Link (Reddit/Insta)":
    link_input = st.text_input("Paste Link (Reddit) or Text:", placeholder="https://www.reddit.com/r/...")
    
    col_a, col_b = st.columns(2)
    with col_a:
        analyze_link_btn = st.button("🔗 ANALYZE REDDIT LINK")
    with col_b:
        with st.expander("📸 Analyze Instagram/Other Text"):
            raw_text = st.text_area("Paste comments from Instagram here:")
            analyze_text_btn = st.button("Analyze Pasted Text")

    if analyze_link_btn and link_input:
        if "reddit.com" in link_input:
            with st.status("🔗 Fetching Comments...", expanded=True) as status:
                df = scraper.fetch_reddit_comments(link_input)
                if df is not None and not df.empty:
                    st.write("✅ Comments Extracted.")
                    df = analyzer.run_analysis(df)
                    status.update(label="✅ Success!", state="complete", expanded=False)
        else:
            st.error("⚠️ Use the paste box for non-Reddit links.")

    if analyze_text_btn and raw_text:
        import pandas as pd
        manual_data = [{"title": "Manual Input", "body": line, "score": 0, "num_comments": 0} for line in raw_text.split('\n') if line.strip()]
        df = pd.DataFrame(manual_data)
        df = analyzer.run_analysis(df)
        st.success(f"✅ Analyzed {len(df)} manual entries.")

# --- DASHBOARD ---
if df is not None and not df.empty:
    
    # 1. CALCULATE BUY SCORE
    pos_count = len(df[df['sentiment'] == 'Positive'])
    neu_count = len(df[df['sentiment'] == 'Neutral'])
    neg_count = len(df[df['sentiment'] == 'Negative'])
    total = len(df)

    if total > 0:
        raw_score = ((pos_count + (0.5 * neu_count)) / total) * 100
        buy_score = round(raw_score, 1)
    else:
        buy_score = 0

    # Determine Verdict
    if buy_score >= 80:
        verdict = "🚀 HIGHLY RECOMMENDED"
        color = "green"
        insight_color = "#00CC96"
        target_sentiment = "Positive"
        why_text = "Users love this because:"
    elif buy_score >= 60:
        verdict = "✅ GOOD TO BUY"
        color = "lightgreen"
        insight_color = "#00CC96"
        target_sentiment = "Positive"
        why_text = "Mostly positive feedback regarding:"
    elif buy_score >= 40:
        verdict = "⚠️ MIXED FEELINGS / RISKY"
        color = "orange"
        insight_color = "#FFA15A"
        target_sentiment = "Negative" # Focus on risks for mixed items
        why_text = "Be careful! Users are complaining about:"
    else:
        verdict = "🛑 NOT RECOMMENDED / AVOID"
        color = "red"
        insight_color = "#EF553B"
        target_sentiment = "Negative"
        why_text = "Major warnings detected regarding:"

    # 2. DISPLAY BUY METER
    st.markdown(f"## 🤖 AI Verdict: <span style='color:{color}'>{verdict}</span>", unsafe_allow_html=True)
    
    col_score, col_bar = st.columns([1, 3])
    with col_score:
        st.metric("Buy Confidence", f"{buy_score}%")
    with col_bar:
        st.write("###") 
        st.progress(buy_score / 100)
    
    # --- NEW: INTELLIGENT REASONING SECTION ---
    st.markdown("---")
    st.subheader(f"🔍 Why? {why_text}")
    
    # Filter for the relevant sentiment (Pros if Good, Cons if Bad)
    target_df = df[df['sentiment'] == target_sentiment]
    
    # If we don't have enough target data, fallback to Neutral or opposite
    if target_df.empty:
        target_df = df 

    if not target_df.empty:
        # A. Extract Keywords
        keywords = extract_top_keywords(target_df['body'].tolist())
        kw_string = ", ".join([f"**{k[0]}**" for k in keywords])
        
        # B. Get Top Upvoted Comment (The "Proof")
        top_review = target_df.sort_values(by='score', ascending=False).iloc[0]
        review_text = top_review['body']
        if len(review_text) > 300: review_text = review_text[:300] + "..."

        # C. Display
        with st.container():
            st.markdown(f"""
            <div class="insight-box" style="border-left: 5px solid {insight_color}">
                <h4>🔑 Common Themes: {kw_string}</h4>
                <p><i>" {review_text} "</i></p>
                <p><b>— Top Rated {target_sentiment} Review</b> (Score: {top_review['score']})</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Not enough data to extract specific reasons.")

    st.markdown("---")

    # 3. METRICS ROW
    st.markdown("### 📊 Market Breakdown")
    c1, c2, c3, c4 = st.columns(4) 
    c1.metric("Total Data Points", total)
    c2.metric("Positive Vibes", f"{pos_count} 😄")
    c3.metric("Neutral / Okay", f"{neu_count} 😐")
    c4.metric("Critical Feedback", f"{neg_count} 😡")

    # 4. CHARTS
    tab1, tab2 = st.tabs(["📈 Sentiment Charts", "📝 Raw Data"])
    
    with tab1:
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            fig_pie = px.pie(df, names='sentiment', hole=0.5, 
                             color='sentiment', title="Community Mood",
                             color_discrete_map={'Positive':'#00CC96', 'Negative':'#EF553B', 'Neutral':'#636EFA'})
            st.plotly_chart(fig_pie, use_container_width=True)
        with col_chart2:
            fig_bar = px.bar(df, x='sentiment', color='sentiment', title="Volume by Sentiment",
                             color_discrete_map={'Positive':'#00CC96', 'Negative':'#EF553B', 'Neutral':'#636EFA'})
            st.plotly_chart(fig_bar, use_container_width=True)

    with tab2:
        st.dataframe(df[['body', 'sentiment', 'score']], use_container_width=True)