
import streamlit as st
import pandas as pd
import pickle
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------- Helper: Clean text ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------- Load cleaned dataset ----------
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_medicine_dataset.csv")
    return df

df = load_data()

# ---------- Load TF-IDF + Cosine Similarity model ----------
@st.cache_resource
def build_similarity_model(data):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(data['Uses'].fillna("").apply(clean_text))
    return vectorizer, tfidf_matrix

vectorizer_sim, tfidf_matrix_sim = build_similarity_model(df)


# ---------- Load Logistic Regression model ----------
@st.cache_resource
def load_lr_model():
    with open("recommend_lr.pkl", "rb") as f:
        data = pickle.load(f)
    return data["model"], data["vectorizer"]

lr_model, tfidf_vectorizer = load_lr_model()


# ---------- TF-IDF Similarity Function ----------
def recommend(symptom, top_n=5):
    cleaned = clean_text(symptom)
    vec = vectorizer_sim.transform([cleaned])
    sims = cosine_similarity(vec, tfidf_matrix_sim).flatten()
    top_indices = sims.argsort()[::-1][:top_n]
    results = df.iloc[top_indices].copy()
    results["Medicine Score"] = sims[top_indices]
    return results

# ---------- Logistic Regression Prediction ----------
def predict_relevance_ml(symptom_text):
    cleaned = clean_text(symptom_text)
    vec = tfidf_vectorizer.transform([cleaned])
    prob = lr_model.predict_proba(vec)[0][1]
    pred = lr_model.predict(vec)[0]
    return pred, prob

# ---------- Streamlit App ----------
st.title("💊 Smart Medicine Recommendation System")

# Sidebar for mode selection
st.sidebar.title("Recommendation Mode")
mode = st.sidebar.radio(
    "Select method:",
    ("TF-IDF Similarity", "Logistic Regression (ML)")
)

# Main Input
user_input = st.text_input("Enter symptom or condition:")

if st.button("Recommend") and user_input.strip() != "":
    if mode == "TF-IDF Similarity":
        st.subheader("Top Recommendations (TF-IDF Similarity)")
        recs = recommend(user_input)
        for i, row in recs.iterrows():
            with st.expander(f"{row['Name']} — Score {row['Medicine Score']:.2f}"):
                st.write(f"**Manufacturer:** {row.get('manufacturer', 'N/A')}")
                st.write(f"**Uses:** {row.get('Uses', 'N/A')}")
                st.write(f"**Composition:** {row.get('Composition', 'N/A')}")
    else:
        st.subheader("ML Model Prediction")
        pred, prob = predict_relevance_ml(user_input)
        if pred == 1:
            st.success(f"✅ This symptom is **likely relevant** to one or more medicines. (Confidence: {prob:.2f})")
        else:
            st.warning(f"⚠️ This symptom is **unlikely relevant** based on ML prediction. (Confidence: {prob:.2f})")
