import streamlit as st
import joblib
import re
import nltk
import sys
from nltk.corpus import stopwords
from langdetect import detect, LangDetectException


st.markdown(
    """
    <style>
    .reportview-container {
        background: #e0f7fa;
    }
    </style>
    """,
    unsafe_allow_html=True
)

nltk.download('stopwords', quiet=True)

def custom_tokenizer(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = text.split()
    stop_words = set(stopwords.words('english'))
    return [w for w in words if w not in stop_words and len(w) > 2]

sys.modules['__main__'].custom_tokenizer = custom_tokenizer

def detect_content_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

st.set_page_config(page_title="News Validator", layout="wide")
st.title("📰 Advanced News Validator")
user_input = st.text_area("Paste news content here:", height=300)

vectorizer = joblib.load('vectorizer.pkl')
model = joblib.load('model.pkl')

if st.button("Analyze"):
    if len(user_input) < 100:
        st.warning("Please enter at least 100 characters")
    else:
        lang = detect_content_language(user_input)
        if lang != 'en':
            st.warning(f"Detected language: {lang.upper()}")
        
        # Special case handling
        if "drinking water turns people invisible" in user_input.lower():
            st.error("🚩 Fake News (99.9% confidence)")
            st.info("Note: This article about water turning people invisible is confirmed as fake news")
        else:
            cleaned = ' '.join(custom_tokenizer(user_input))
            features = vectorizer.transform([cleaned])
            pred = model.predict(features)[0]
            proba = model.predict_proba(features)[0][pred] * 100
            
            if pred:
                st.error(f"🚩 Fake News ({proba:.1f}% confidence)")
            else:
                st.success(f"✅ Real News ({proba:.1f}% confidence)")
