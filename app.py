# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import io
import requests
from io import BytesIO

# --- optional NLP libs used in your preprocessing (ensure included in requirements.txt) ---
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
TARGET_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# default file names (if you put them inside the repo)
DEFAULT_MODEL_PATH = "naive_bayes_model.pkl"
DEFAULT_VECT_PATH = "tfidf_vectorizer.pkl"
SAMPLE_SUB_PATH = "sample_submission.csv"  # optional

# -----------------------------------------------------------------------------
# HELPERS: load model & vectorizer (local or from raw github)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def download_bytes_from_url(url: str) -> bytes:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.content

@st.cache_resource
def load_artifacts(model_path: str = DEFAULT_MODEL_PATH, vect_path: str = DEFAULT_VECT_PATH,
                   model_url: str = None, vect_url: str = None):
    """
    Load model & vectorizer either from local file paths (when deployed from repo)
    or from remote raw URLs (raw.githubusercontent.com).
    Returns (model, vectorizer).
    """
    # load vectorizer
    if vect_url:
        b = download_bytes_from_url(vect_url)
        vect = joblib.load(BytesIO(b))
    else:
        vect = joblib.load(vect_path)

    # load model
    if model_url:
        b = download_bytes_from_url(model_url)
        model = joblib.load(BytesIO(b))
    else:
        model = joblib.load(model_path)

    return model, vect

# -----------------------------------------------------------------------------
# Preprocessing - should match training preprocessing
# -----------------------------------------------------------------------------
@st.cache_resource
def setup_nlp():
    try:
        nltk.data.find("corpora/stopwords")
    except Exception:
        nltk.download("stopwords")
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    return stemmer, stop_words

stemmer, STOP_WORDS = setup_nlp()

def clean_text(text: str) -> str:
    text = str(text)
    # basic normalization: remove urls, mentions, hashtags, punctuation, lowercase
    text = re.sub(r"(https?://\S+|www\.\S+)", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = text.lower()
    tokens = text.split()
    cleaned = []
    for t in tokens:
        if not t.isalpha():
            continue
        if t in STOP_WORDS:
            continue
        try:
            cleaned.append(stemmer.stem(t))
        except RecursionError:
            continue
    return " ".join(cleaned)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def predict_proba_and_labels(model, vect, texts, thresholds):
    """
    texts: list of raw texts (strings)
    thresholds: dict label->threshold (0-1)
    returns: probs (ndarray), preds (ndarray 0/1)
    """
    cleaned = [clean_text(t) for t in texts]
    X = vect.transform(cleaned)
    # OneVsRestClassifier with MultinomialNB supports predict_proba
    try:
        probs = model.predict_proba(X)
    except Exception:
        # fallback to predict (0/1)
        preds = model.predict(X)
        probs = preds.astype(float)
    # convert to binary using thresholds
    preds_bin = np.zeros_like(probs, dtype=int)
    for i, col in enumerate(TARGET_COLS):
        thr = thresholds.get(col, 0.5)
        preds_bin[:, i] = (probs[:, i] >= thr).astype(int)
    return probs, preds_bin, cleaned

def top_tokens_for_class(model, vect, class_index, top_n=10):
    """
    For MultinomialNB wrapped in OneVsRestClassifier:
      estimator = model.estimators_[i] (MultinomialNB)
      estimator.feature_log_prob_[1] -> log prob of features for positive class
    vect.get_feature_names_out() -> feature tokens
    returns list of (token, score)
    """
    try:
        estimators = model.estimators_
    except Exception:
        return []
    if class_index >= len(estimators):
        return []
    mb = estimators[class_index]
    # feature_log_prob_ shape: (n_classes_binary, n_features) -> choose positive class (1)
    if hasattr(mb, "feature_log_prob_"):
        arr = mb.feature_log_prob_
        # If binary: arr.shape[0] may be 2 => choose row 1 for positive class
        if arr.shape[0] == 2:
            scores = arr[1]
        else:
            # if single row, use it directly
            scores = arr[0]
        feature_names = vect.get_feature_names_out()
        top_idx = np.argsort(scores)[-top_n:][::-1]
        return [(feature_names[i], float(scores[i])) for i in top_idx]
    return []

# -----------------------------------------------------------------------------
# STREAMLIT UI
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Content Moderation System", layout="wide",
                   initial_sidebar_state="expanded")
# Internal settings (not visible to user)
model_path = DEFAULT_MODEL_PATH
vect_path = DEFAULT_VECT_PATH
model_url = ""
vect_url = ""

thresholds = {c: 0.5 for c in TARGET_COLS}  # default thresholds

# main
st.title("🚦 Content Moderation System")
st.markdown(
    """
    Professional, simple interface for your trained Naive Bayes content moderation model.
    *Supports single-text prediction and batch CSV predictions .*
    """
)

# Load model & vectorizer
with st.spinner("Loading model & vectorizer..."):
    try:
        model, vect = load_artifacts(model_path=model_path, vect_path=vect_path,
                                     model_url=(model_url or None), vect_url=(vect_url or None))
    except Exception as e:
        st.error(f"Failed loading model/vectorizer: {e}")
        st.stop()

st.success("Model & vectorizer loaded ✅")

# Layout: two columns
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Single comment classification")
    text = st.text_area("Enter a comment to classify", height=150, placeholder="Type or paste comment here...")
    if st.button("Predict single comment"):
        if not text.strip():
            st.warning("Please enter a comment first.")
        else:
            with st.spinner("Predicting..."):
                probs, preds, cleaned = predict_proba_and_labels(model, vect, [text], thresholds)
            st.markdown("**Preprocessing (applied):**")
            st.code(cleaned[0])
            # show probabilities
            prob_row = probs[0]
            dfp = pd.DataFrame([prob_row], columns=TARGET_COLS).T
            dfp.columns = ["probability"]
            st.subheader("Predicted probabilities")
            st.table(dfp)

            st.subheader("Predicted labels (thresholded)")
            res = {TARGET_COLS[i]: int(preds[0, i]) for i in range(len(TARGET_COLS))}
            st.json(res)

            # show bar chart
            st.subheader("Probabilities chart")
            st.bar_chart(pd.DataFrame(prob_row.reshape(1,-1), columns=TARGET_COLS))

            # explanation: top tokens per predicted positive class
            st.subheader("Top tokens for each predicted class")
            for i, label in enumerate(TARGET_COLS):
                if preds[0, i] == 1:
                    top = top_tokens_for_class(model, vect, i, top_n=10)
                    st.markdown(f"**{label}**")
                    if top:
                        top_df = pd.DataFrame(top, columns=["token", "log_prob"])
                        st.table(top_df)
                    else:
                        st.write("No explanation available for this class.")

with col2:
    st.subheader("Batch prediction (CSV)")
    uploaded = st.file_uploader("Upload CSV with `id` and `comment_text`", type=["csv"])
    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            batch_df = None
        if batch_df is not None:
            if "id" not in batch_df.columns or "comment_text" not in batch_df.columns:
                st.error("CSV must contain `id` and `comment_text` columns.")
            else:
                st.write("Preview:")
                st.dataframe(batch_df.head())

                if st.button("Run batch prediction"):
                    with st.spinner("Running predictions..."):
                        texts = batch_df["comment_text"].astype(str).tolist()
                        probs, preds_bin, _ = predict_proba_and_labels(model, vect, texts, thresholds)
                        preds_df = pd.DataFrame(preds_bin, columns=TARGET_COLS)
                        out_df = pd.concat([batch_df[["id"]].reset_index(drop=True), preds_df], axis=1)
                        csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                    st.success("Batch prediction done.")
                    st.download_button("Download submission.csv", data=csv_bytes, file_name="submission.csv",
                                       mime="text/csv")
                    st.dataframe(out_df.head())

# Footer: sample submission (if available)
st.markdown("---")
st.subheader("Sample submission")
try:
    sample = pd.read_csv(SAMPLE_SUB_PATH)
    st.write("Sample submission preview:")
    st.dataframe(sample.head())
except Exception:
    st.info("No sample_submission.csv")


