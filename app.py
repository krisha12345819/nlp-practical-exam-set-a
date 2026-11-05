"""
CineSense - Movie Review Sentiment Analyser
Practical Exam (Set A) - Step 6: Streamlit deployment
Run with:  streamlit run app.py
"""

import re
from pathlib import Path

import joblib
import nltk
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="CineSense | Review Sentiment Analyser",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).parent / "sentiment_pipeline.pkl"

EXAMPLES = {
    "😊 Clearly positive": (
        "An absolutely wonderful film! The acting was excellent, the story was "
        "beautiful and I loved every single scene. A perfect movie."
    ),
    "😞 Clearly negative": (
        "This was the worst movie I have ever seen. Boring plot, terrible acting "
        "and a complete waste of time. Awful."
    ),
    "🤔 Tricky (negation)": (
        "The movie was not good at all, but the music was not bad."
    ),
}

POS_MEANING = {
    "NN": "Noun (singular)", "NNS": "Noun (plural)", "NNP": "Proper noun",
    "VB": "Verb (base)", "VBD": "Verb (past)", "VBG": "Verb (gerund)",
    "VBN": "Verb (past participle)", "VBZ": "Verb (3rd person)",
    "VBP": "Verb (present)", "JJ": "Adjective", "JJR": "Adjective (comparative)",
    "JJS": "Adjective (superlative)", "RB": "Adverb", "DT": "Determiner",
    "IN": "Preposition / conjunction", "CC": "Coordinating conjunction",
    "PRP": "Personal pronoun", "PRP$": "Possessive pronoun", "TO": "'to'",
    "MD": "Modal verb", "CD": "Number", "WDT": "Wh-determiner",
}

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
    #MainMenu, footer { visibility: hidden; }
    .block-container { padding-top: 1.6rem; max-width: 820px; }

    .hero {
        background: linear-gradient(135deg, #7F00FF 0%, #E100FF 55%, #FF5F6D 100%);
        padding: 2.2rem 1.6rem; border-radius: 22px; text-align: center;
        color: white; box-shadow: 0 12px 34px rgba(127,0,255,.28);
        margin-bottom: 1.4rem;
    }
    .hero h1 { margin: 0; font-size: 2.5rem; font-weight: 700; letter-spacing: .5px; }
    .hero p  { margin: .5rem 0 0; font-size: 1.02rem; opacity: .95; }
    .pill {
        display: inline-block; margin-top: .9rem; padding: .25rem .85rem;
        background: rgba(255,255,255,.2); border-radius: 999px; font-size: .78rem;
        backdrop-filter: blur(4px);
    }

    .result-card {
        border-radius: 20px; padding: 1.6rem 1.4rem; text-align: center;
        margin: 1.1rem 0 .6rem; color: white;
        animation: pop .45s ease-out;
    }
    .result-card.pos { background: linear-gradient(135deg, #11998e, #38ef7d); box-shadow: 0 10px 28px rgba(17,153,142,.35); }
    .result-card.neg { background: linear-gradient(135deg, #cb2d3e, #ef473a); box-shadow: 0 10px 28px rgba(203,45,62,.35); }
    .result-card .emoji { font-size: 3.4rem; line-height: 1; }
    .result-card .label { font-size: 2rem; font-weight: 700; margin-top: .3rem; }
    .result-card .conf  { font-size: 1.05rem; opacity: .95; }
    @keyframes pop { from {transform: scale(.92); opacity: 0;} to {transform: scale(1); opacity: 1;} }

    .stButton > button {
        width: 100%; border: 0; border-radius: 14px; padding: .7rem 1rem;
        font-weight: 600; font-size: 1.05rem; color: white;
        background: linear-gradient(90deg, #7F00FF, #E100FF);
        transition: transform .15s ease, box-shadow .15s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px); color: white;
        box-shadow: 0 8px 20px rgba(127,0,255,.35);
    }
    .stTextArea textarea { border-radius: 14px; font-size: 1rem; }
    div[data-testid="stMetric"] {
        background: rgba(127,0,255,.07); border: 1px solid rgba(127,0,255,.18);
        padding: .7rem 1rem; border-radius: 14px;
    }
    .foot { text-align: center; opacity: .6; font-size: .82rem; margin-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Resources
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading NLP resources...")
def load_nltk():
    for pkg in [
        "punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4",
        "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
    ]:
        nltk.download(pkg, quiet=True)
    return set(stopwords.words("english")), WordNetLemmatizer()


@st.cache_resource(show_spinner="Loading sentiment model...")
def load_pipeline():
    return joblib.load(MODEL_PATH)


STOP_WORDS, LEMMATIZER = load_nltk()


def clean_text(text: str) -> str:
    """Same preprocessing used when the pipeline was trained (Steps 3 & 4)."""
    text = re.sub(r"<.*?>", "", text)             # remove HTML tags
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)    # remove URLs
    text = re.sub(r"[^a-z\s]", "", text)          # keep letters + spaces
    words = word_tokenize(text)
    words = [w for w in words if w not in STOP_WORDS]
    words = [LEMMATIZER.lemmatize(w) for w in words]
    return " ".join(words)


def top_influential_words(pipeline, cleaned: str, predicted: int, k: int = 3):
    """Words whose TF-IDF x coefficient pushed the prediction the most."""
    vec = pipeline.named_steps["vectorizer"]
    clf = pipeline.named_steps["classifier"]
    row = vec.transform([cleaned]).toarray()[0]
    contrib = row * clf.coef_[0]
    names = vec.get_feature_names_out()
    idx = np.nonzero(row)[0]
    if len(idx) == 0:
        return pd.DataFrame(columns=["Word", "Impact"])
    sign = 1 if predicted == 1 else -1
    scored = sorted(idx, key=lambda i: sign * contrib[i], reverse=True)[:k]
    return pd.DataFrame(
        {"Word": [names[i] for i in scored], "Impact": [abs(contrib[i]) for i in scored]}
    )


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
if "review_text" not in st.session_state:
    st.session_state["review_text"] = ""


def use_example(text: str):
    st.session_state["review_text"] = text


with st.sidebar:
    st.markdown("## 🎬 CineSense")
    st.caption("Review analytics for an OTT platform")
    st.markdown("---")
    st.markdown("### 💡 Example reviews")
    st.caption("Read them here, or click to load one into the box.")
    for title, text in EXAMPLES.items():
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.write(text)
            st.button("Use this review", key=f"btn_{title}", on_click=use_example, args=(text,))
    st.markdown("---")
    st.markdown("### ℹ️ About this project")
    st.info(
        "Built for the NLP Practical Exam (Set A). The model is a **TF-IDF + "
        "Logistic Regression** pipeline trained on 5,000 balanced IMDb reviews "
        "(2,500 positive / 2,500 negative). Text is cleaned with RegEx, "
        "tokenisation, stop-word removal and lemmatisation before prediction."
    )

# ----------------------------------------------------------------------------
# Main page
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🎬 CineSense</h1>
        <p>Paste a movie review and instantly discover whether the viewer loved it or not.</p>
        <span class="pill">NLP • TF-IDF • Logistic Regression</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error(
        "`sentiment_pipeline.pkl` was not found next to `app.py`. "
        "Copy the file saved in Step 4 into the same folder and refresh."
    )
    st.stop()

pipeline = load_pipeline()

review = st.text_area(
    "✍️ Your review",
    key="review_text",
    height=170,
    placeholder="e.g. The story was gripping and the acting was superb...",
)
st.caption(f"{len(review.split())} words • {len(review)} characters")

if st.button("🔍 Analyse Sentiment"):
    if not review.strip():
        st.warning("Please enter a review")
    else:
        cleaned = clean_text(review)
        if not cleaned.strip():
            st.warning("No meaningful English words found. Try a longer review.")
        else:
            proba = pipeline.predict_proba([cleaned])[0]
            pred = int(pipeline.predict([cleaned])[0])
            confidence = float(proba.max()) * 100
            is_pos = pred == 1

            st.markdown(
                f"""
                <div class="result-card {'pos' if is_pos else 'neg'}">
                    <div class="emoji">{'😊' if is_pos else '😞'}</div>
                    <div class="label">{'Positive' if is_pos else 'Negative'}</div>
                    <div class="conf">Confidence: {confidence:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(min(confidence / 100, 1.0))

            c1, c2, c3 = st.columns(3)
            c1.metric("Positive probability", f"{proba[1] * 100:.1f}%")
            c2.metric("Negative probability", f"{proba[0] * 100:.1f}%")
            c3.metric("Words analysed", len(cleaned.split()))

            # ---- Bonus 1: top 3 influential words ----
            st.markdown("#### 🔑 Top 3 words that pushed the prediction")
            top = top_influential_words(pipeline, cleaned, pred)
            if top.empty:
                st.caption("None of the words were in the model vocabulary.")
            else:
                color = "#11998e" if is_pos else "#cb2d3e"
                chart = (
                    alt.Chart(top)
                    .mark_bar(cornerRadiusEnd=8, color=color)
                    .encode(
                        x=alt.X("Impact:Q", title="Influence on prediction"),
                        y=alt.Y("Word:N", sort="-x", title=None),
                        tooltip=["Word", alt.Tooltip("Impact:Q", format=".3f")],
                    )
                    .properties(height=140)
                )
                st.altair_chart(chart, use_container_width=True)

            # ---- Bonus 2: POS tags ----
            with st.expander("🧩 Part-of-Speech tags of your review"):
                tags = pos_tag(word_tokenize(review))
                pos_df = pd.DataFrame(tags, columns=["Word", "POS Tag"])
                pos_df["Meaning"] = pos_df["POS Tag"].map(POS_MEANING).fillna("Other")
                st.dataframe(pos_df, use_container_width=True, hide_index=True)

            with st.expander("🧼 See the cleaned text the model received"):
                st.code(cleaned or "(empty)", language="text")

st.markdown(
    '<div class="foot">Made with ❤️ using Streamlit & scikit-learn • NLP Practical Exam – Set A</div>',
    unsafe_allow_html=True,
)
