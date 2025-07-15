import streamlit as st
import pandas as pd
import pickle
import re
from sklearn.metrics.pairwise import cosine_similarity

# ------------------- Helper Functions -------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def calculate_medicine_score(excellent, average, poor):
    score = (excellent * 0.06) + (average * 0.03) + (poor * 0.01)
    return round(score, 2)

def recommend_medicines(symptom_text, vectorizer, matrix, df):
    cleaned = clean_text(symptom_text)
    input_vec = vectorizer.transform([cleaned])
    sim_scores = cosine_similarity(matrix, input_vec).flatten()
    top_indices = sim_scores.argsort()[::-1][:10]

    recs = df.iloc[top_indices].copy()
    recs["Medicine Score"] = recs.apply(
        lambda row: calculate_medicine_score(row["Excellent"], row["Average"], row["Poor"]), axis=1
    )

    recs["final_score"] = 0.7 * sim_scores[top_indices] + 0.3 * recs["Medicine Score"] / 6
    recs = recs.sort_values("final_score", ascending=False).head(6)
    recs["Similarity %"] = [round(sim_scores[i]*100, 1) for i in recs.index]
    return recs

# ------------------- Load Model -------------------

with open("recommend.pkl", "rb") as f:
    model = pickle.load(f)

vectorizer = model["tfidf_vectorizer_uses"]
matrix = model["tfidf_matrix_uses"]
df = model["clean_df"]

# ------------------- Streamlit UI -------------------

st.set_page_config(page_title="Medicine Recommender", layout="wide")

st.markdown("""
    <h1 style='text-align: center;'>💊 AI Medicine Recommendation System</h1>
    <div style="background-color:#fffbe6;
                padding:10px;
                border:1px solid #ffec99;
                border-radius:6px;
                color:#333;">
    🛑 <strong>Disclaimer:</strong> This tool suggests over-the-counter (OTC) medicines based on textual similarity.
    It is <u>not</u> a substitute for professional medical advice. Always consult a doctor.
    </div>
""", unsafe_allow_html=True)

st.write("")

user_input = st.text_area("📝 Describe your symptoms (in a sentence or two):", height=100)

if st.button("🔍 Recommend Medicines"):

    if not user_input.strip():
        st.warning("Please enter some symptoms first.")
        st.stop()

    results = recommend_medicines(user_input, vectorizer, matrix, df)

    st.success("Top medicine suggestions:")
    for _, row in results.iterrows():
        with st.expander(f"{row['Name']} (Score: {row['Medicine Score']}, Match: {row['Similarity %']}%)"):
            st.write(f"**Manufacturer**: {row.get('manufacturer', 'N/A')}")
            st.write(f"**Uses**: {row['Uses']}")
            st.write(f"**Side Effects**: {row.get('Side_effects', 'N/A')}")
            st.write(f"**How to Use**: {row.get('How_to_use', 'N/A')}")
            st.markdown(f"[📦 Product Page]({row.get('Links-href', '#')})")

# Optional: footer
st.markdown("<hr><center><small>Developed for Final Year Project — University of Greenwich</small></center>", unsafe_allow_html=True)
