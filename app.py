import streamlit as st
import joblib
import re
import html

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Movie Review Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.html("""
<style>

.stApp {
    background: linear-gradient(135deg, #070d1a 0%, #0c1425 55%, #101b31 100%);
    color: #f5f7ff;
}

.main .block-container {
    max-width: 1180px;
    padding-top: 35px;
    padding-bottom: 30px;
}

/* Header */

.app-title {
    text-align: center;
    font-size: 48px;
    font-weight: 750;
    margin-bottom: 5px;
    color: #f5f7ff;
    letter-spacing: -1px;
}

.app-title .highlight {
    color: #a56cff;
}

.app-subtitle {
    text-align: center;
    color: #b9c4d8;
    font-size: 18px;
    margin-bottom: 35px;
}


/* Section */

.section-box {
    background: rgba(12, 23, 42, 0.92);
    border: 1px solid #263e63;
    border-radius: 16px;
    padding: 26px;
    margin-top: 18px;
    margin-bottom: 25px;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.22);
}

.section-title {
    font-size: 27px;
    font-weight: 700;
    color: #f5f7ff;
    margin-bottom: 20px;
}

.review-label {
    color: #c5d0e3;
    font-size: 17px;
    margin-bottom: 8px;
}


/* Text area */

textarea {
    background-color: #162845 !important;
    color: #f1f5ff !important;
    border: 1px solid #31578a !important;
    border-radius: 12px !important;
    font-size: 17px !important;
}

textarea:focus {
    border: 1px solid #647cff !important;
    box-shadow: 0 0 0 1px #647cff !important;
}


/* Analyze button */

.stButton > button {
    background: linear-gradient(135deg, #8b4dff, #176dff) !important;
    color: white !important;
    border: none !important;
    border-radius: 11px !important;
    padding: 13px 30px !important;
    font-size: 17px !important;
    font-weight: 650 !important;
    margin-top: 8px;
}

.stButton > button:hover {
    box-shadow: 0 8px 22px rgba(78, 101, 255, 0.35);
    transform: translateY(-2px);
}


/* Your review */

.review-title {
    color: #d6deed;
    font-size: 17px;
    font-weight: 600;
    margin-bottom: 9px;
}

.review-display {
    background: linear-gradient(135deg, #152b4d, #1b365f);
    border: 1px solid #1685ca;
    border-radius: 12px;
    padding: 17px 20px;
    color: #edf5ff;
    font-size: 17px;
    line-height: 1.6;
    margin-bottom: 24px;
}


/* Result cards */

.result-card {
    border-radius: 14px;
    min-height: 150px;
    padding: 22px;
}

.sentiment-card {
    border: 1px solid #11c995;
    background: linear-gradient(
        135deg,
        rgba(7, 72, 65, 0.72),
        rgba(7, 45, 44, 0.85)
    );
}

.sarcasm-card {
    border: 1px solid #a345ff;
    background: linear-gradient(
        135deg,
        rgba(72, 27, 113, 0.72),
        rgba(43, 20, 75, 0.85)
    );
}

.probability-card {
    border: 1px solid #148cff;
    background: linear-gradient(
        135deg,
        rgba(17, 63, 111, 0.72),
        rgba(15, 43, 78, 0.85)
    );
}

.card-label {
    color: #cbd6e8;
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 16px;
}

.card-value {
    font-size: 30px;
    font-weight: 750;
}

.positive {
    color: #42e4a5;
}

.negative {
    color: #ff727c;
}

.sarcastic {
    color: #c47bff;
}

.non-sarcastic {
    color: #71c4ff;
}

.probability {
    color: #5bb6ff;
}


/* Interpretation */

.interpretation {
    border-top: 1px solid #2b3e5c;
    margin-top: 28px;
    padding-top: 22px;
}

.interpretation-title {
    color: #f2f5ff;
    font-size: 21px;
    font-weight: 700;
    margin-bottom: 12px;
}

.interpretation-text {
    color: #c5cfdf;
    font-size: 16px;
    line-height: 1.8;
}


/* Footer */

.footer {
    text-align: center;
    color: #73829b;
    font-size: 14px;
    padding-top: 25px;
    padding-bottom: 10px;
}

</style>
""")


# =========================================================
# LOAD MODELS
# =========================================================

sentiment_model = joblib.load("sentiment_model.pkl")
sentiment_tfidf = joblib.load("sentiment_tfidf.pkl")

sarcasm_model = joblib.load("sarcasm_model.pkl")
sarcasm_tfidf = joblib.load("sarcasm_tfidf.pkl")


# =========================================================
# NLTK SETUP
# =========================================================

stop_words = set(stopwords.words("english"))


# =========================================================
# SENTIMENT PREPROCESSING
# =========================================================

def preprocess_text(text):

    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Keep only alphabets and spaces
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Tokenization
    words = word_tokenize(text)

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    return " ".join(words)


# =========================================================
# SARCASM PREPROCESSING
# =========================================================

def preprocess_sarcasm(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove punctuation and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# =========================================================
# SENTIMENT PREDICTION
# =========================================================

def predict_sentiment(review):

    processed = preprocess_text(review)

    vectorized = sentiment_tfidf.transform([processed])

    prediction = sentiment_model.predict(vectorized)[0]

    if prediction == 1:
        return "Positive"
    else:
        return "Negative"


# =========================================================
# SARCASM PREDICTION
# =========================================================

def predict_sarcasm(review):

    processed = preprocess_sarcasm(review)

    vectorized = sarcasm_tfidf.transform([processed])

    prediction = sarcasm_model.predict(vectorized)[0]

    probability = sarcasm_model.predict_proba(vectorized)[0][1]

    if prediction == 1:
        result = "Sarcastic"
    else:
        result = "Non-Sarcastic"

    return result, probability


# =========================================================
# HEADER
# =========================================================

st.html("""
<div class="app-title">
    🎬 Movie Review <span class="highlight">Analyzer</span>
</div>

<div class="app-subtitle">
    Analyze movie reviews using Sentiment Analysis and Sarcasm Detection
</div>
""")


# =========================================================
# INPUT SECTION
# =========================================================

st.html("""
<div class="section-box">

    <div class="section-title">
        📝 Enter Movie Review
    </div>

    <div class="review-label">
        Write your movie review below:
    </div>

</div>
""")


review = st.text_area(
    "Movie Review",
    placeholder="Write your movie review here...",
    height=140,
    label_visibility="collapsed"
)


# =========================================================
# ANALYZE BUTTON
# =========================================================

analyze = st.button(
    "🔍  Analyze Review",
    use_container_width=False
)


# =========================================================
# ANALYSIS
# =========================================================

if analyze:

    if not review.strip():

        st.warning("⚠️ Please enter a movie review first.")

    else:

        # ---------------------------------------------
        # MODEL PREDICTIONS
        # ---------------------------------------------

        sentiment = predict_sentiment(review)

        sarcasm, sarcasm_probability = predict_sarcasm(review)

        sarcasm_percentage = sarcasm_probability * 100


        # ---------------------------------------------
        # ESCAPE USER TEXT FOR SAFE HTML DISPLAY
        # ---------------------------------------------

        safe_review = html.escape(review)


        # ---------------------------------------------
        # SENTIMENT STYLE
        # ---------------------------------------------

        if sentiment == "Positive":
            sentiment_class = "positive"
            sentiment_icon = "😊"
        else:
            sentiment_class = "negative"
            sentiment_icon = "☹️"


        # ---------------------------------------------
        # SARCASM STYLE
        # ---------------------------------------------

        if sarcasm == "Sarcastic":
            sarcasm_class = "sarcastic"
            sarcasm_icon = "🎭"
        else:
            sarcasm_class = "non-sarcastic"
            sarcasm_icon = "💬"


        # ---------------------------------------------
        # INTERPRETATION
        # ---------------------------------------------

        if sentiment == "Positive":

            sentiment_message = (
                "The review expresses a positive opinion about the movie."
            )

        else:

            sentiment_message = (
                "The review expresses a negative opinion about the movie."
            )


        if sarcasm == "Sarcastic":

            sarcasm_message = (
                "The model detected sarcastic language in the review."
            )

        else:

            sarcasm_message = (
                "The model did not detect strong sarcastic language."
            )


        # =================================================
        # RESULT SECTION
        # =================================================

        st.html("""
<div class="section-box">

    <div class="section-title">
        📊 Analysis Result
    </div>

    <div class="review-title">
        💬 Your Review
    </div>
""")

        st.html(
            f"""
<div class="review-display">
    {safe_review}
</div>
"""
        )


        # =================================================
        # RESULT CARDS
        # =================================================

        col1, col2, col3 = st.columns(3, gap="medium")


        # ---------------------------------------------
        # SENTIMENT CARD
        # ---------------------------------------------

        with col1:

            st.html(
                f"""
<div class="result-card sentiment-card">

    <div class="card-label">
        {sentiment_icon} &nbsp; Sentiment
    </div>

    <div class="card-value {sentiment_class}">
        {sentiment}
    </div>

</div>
"""
            )


        # ---------------------------------------------
        # SARCASM CARD
        # ---------------------------------------------

        with col2:

            st.html(
                f"""
<div class="result-card sarcasm-card">

    <div class="card-label">
        {sarcasm_icon} &nbsp; Sarcasm
    </div>

    <div class="card-value {sarcasm_class}">
        {sarcasm}
    </div>

</div>
"""
            )


        # ---------------------------------------------
        # PROBABILITY CARD
        # ---------------------------------------------

        with col3:

            st.html(
                f"""
<div class="result-card probability-card">

    <div class="card-label">
        📈 &nbsp; Sarcasm Probability
    </div>

    <div class="card-value probability">
        {sarcasm_percentage:.2f}%
    </div>

</div>
"""
            )


        # =================================================
        # INTERPRETATION
        # =================================================

        st.html(
            f"""
<div class="interpretation">

    <div class="interpretation-title">
        💡 Interpretation
    </div>

    <div class="interpretation-text">
        • {sentiment_message}<br>
        • {sarcasm_message}
    </div>

</div>

</div>
"""
        )


# =========================================================
# FOOTER
# =========================================================

st.html("""
<div class="footer">
    🎬 Movie Review Analyzer
</div>
""")
