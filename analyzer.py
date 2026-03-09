import streamlit as st
import pandas as pd
import numpy as np
import os
from scipy.special import softmax
from deep_translator import GoogleTranslator

# --- CRITICAL: POINT TO THE LOCAL FOLDER ---
# This ensures we use the downloaded file and don't touch the broken cache
MODEL_PATH = "./my_local_model"

@st.cache_resource
def load_ai_model():
    """
    Loads the AI Model from the LOCAL folder.
    """
    print(f"🧠 Loading AI from {MODEL_PATH}...")
    
    # Safety Check: Ensure the user ran the download script
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Error: The folder '{MODEL_PATH}' was not found. Please run 'force_download.py' first!")
        st.stop()
        
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    # Load from the local folder, NOT the internet
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
        return tokenizer, model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

def run_analysis(df):
    """
    Takes a DataFrame, translates text if needed, runs AI sentiment analysis.
    """
    tokenizer, model = load_ai_model()
    encoded_sentiments = []
    
    # Progress Bar configuration
    progress_bar = st.progress(0)
    total_posts = len(df)
    
    # Initialize Translator
    translator = GoogleTranslator(source='auto', target='en')

    print(f"🌍 Starting Analysis for {total_posts} items...")
    
    # --- STEP 1: PREPARE TEXT (Selection & Translation) ---
    text_data = []
    for index, row in df.iterrows():
        body_text = str(row.get('body', ''))
        title_text = str(row.get('title', ''))
        
        # Select the best text (Comment Body > Post Title)
        if len(body_text) > 5 and body_text not in ["[removed]", "[deleted]"]:
            raw_text = body_text[:512]
        else:
            raw_text = title_text[:512]
            
        # Translate Tanglish/Other languages to English
        try:
            # Convert "Padam mokka" -> "Movie is bad"
            translated_text = translator.translate(raw_text)
            if not translated_text: 
                text_data.append(raw_text)
            else:
                text_data.append(translated_text)
        except Exception:
            text_data.append(raw_text) # Fallback if no internet for translator

    # --- STEP 2: AI ANALYSIS ---
    for i, text in enumerate(text_data):
        try:
            # Tokenize and Run AI
            encoded_input = tokenizer(text, return_tensors='pt')
            output = model(**encoded_input)
            scores = output.logits[0].detach().numpy()
            scores = softmax(scores)

            # Find strongest sentiment
            ranking = np.argsort(scores)
            ranking = ranking[::-1]
            
            labels = ['Negative', 'Neutral', 'Positive']
            top_sentiment = labels[ranking[0]]
            
            encoded_sentiments.append(top_sentiment)

        except Exception as e:
            print(f"Error on row {i}: {e}")
            encoded_sentiments.append("Neutral") # Default if error
        
        # Update progress bar
        progress_bar.progress((i + 1) / total_posts)

    progress_bar.empty()
    
    df['sentiment'] = encoded_sentiments
    return df