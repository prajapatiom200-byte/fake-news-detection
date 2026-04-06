import streamlit as st
import pickle
import os
from utils import preprocess_text

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# ----------------------------
# CUSTOM CSS
# ----------------------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
    color: white;
}
.card {
    padding: 20px;
    border-radius: 12px;
    background: #1c1f26;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
}
.result-real {
    color: #00ff9d;
    font-weight: bold;
    font-size: 20px;
}
.result-fake {
    color: #ff4b4b;
    font-weight: bold;
    font-size: 20px;
}
.center {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# LOAD MODELS
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    log_model = pickle.load(open(os.path.join(BASE_DIR, "model_logistic.pkl"), "rb"))
    nb_model = pickle.load(open(os.path.join(BASE_DIR, "model_nb.pkl"), "rb"))
    vectorizer = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))
except:
    st.error("❌ Models not found! Please run train.py first.")
    st.stop()

# ----------------------------
# HEADER
# ----------------------------
st.markdown("<h1 class='center'>📰 Fake News Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p class='center'>AI-based classification using NLP & ML</p>", unsafe_allow_html=True)

# ----------------------------
# INPUT
# ----------------------------
st.markdown("### ✍️ Enter News Content")

user_input = st.text_area(
    "Paste headline or full news",
    height=150,
    placeholder="Example: India wins T20 World Cup 2024..."
)

# Word count display
if user_input:
    st.caption(f"Word count: {len(user_input.split())}")

# ----------------------------
# BUTTON
# ----------------------------
if st.button("🔍 Analyze News"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter some news text")
        st.stop()

    # ----------------------------
    # PREPROCESS
    # ----------------------------
    cleaned = preprocess_text(user_input)
    vectorized = vectorizer.transform([cleaned])

    # ----------------------------
    # PREDICTIONS
    # ----------------------------
    log_pred = log_model.predict(vectorized)[0]
    nb_pred = nb_model.predict(vectorized)[0]

    log_prob = log_model.predict_proba(vectorized)[0].max()
    nb_prob = nb_model.predict_proba(vectorized)[0].max()

    def label(x):
        return "REAL" if x == 1 else "FAKE"

    # ----------------------------
    # DISPLAY PREDICTIONS
    # ----------------------------
    st.markdown("## 🔍 Model Predictions")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Logistic Regression")

        if log_pred == 1:
            st.markdown("<p class='result-real'>REAL ✅</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='result-fake'>FAKE ❌</p>", unsafe_allow_html=True)

        st.write(f"Confidence: {log_prob:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Naive Bayes")

        if nb_pred == 1:
            st.markdown("<p class='result-real'>REAL ✅</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='result-fake'>FAKE ❌</p>", unsafe_allow_html=True)

        st.write(f"Confidence: {nb_prob:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------------
    # SMART FINAL VERDICT
    # ----------------------------
    st.markdown("## 🧠 Final Verdict")

    log_score = log_prob if log_pred == 1 else -log_prob
    nb_score = nb_prob if nb_pred == 1 else -nb_prob

    final_score = log_score + nb_score

    word_count = len(user_input.split())

    # 🔥 Special handling for short news
    if word_count <= 8:
        st.info("⚠️ Short text detected — using optimized prediction")

        final_score = (0.4 * log_score) + (0.6 * nb_score)

    # Final decision
    if final_score > 0:
        st.success("✅ FINAL RESULT: REAL")
    else:
        st.error("❌ FINAL RESULT: FAKE")

    # ----------------------------
    # MODEL PERFORMANCE EXPLANATION
    # ----------------------------
    st.markdown("## 📊 Model Performance Explanation")

    if word_count <= 8:
        st.info(f"""
        This is a short input.

        👉 Naive Bayes performed better because:
        - It works on keywords
        - It handles short text effectively

        Confidence:
        - Logistic: {log_prob:.2f}
        - Naive Bayes: {nb_prob:.2f}
        """)

    elif log_prob > nb_prob:
        st.info(f"""
        Logistic Regression performed better.

        Reason:
        - Better context understanding
        - Higher confidence ({log_prob:.2f} vs {nb_prob:.2f})
        """)

    else:
        st.info(f"""
        Naive Bayes performed better.

        Reason:
        - Strong keyword detection
        - Higher confidence ({nb_prob:.2f} vs {log_prob:.2f})
        """)

    # ----------------------------
    # NOTE
    # ----------------------------
    st.markdown("## 📌 Note")

    st.write("""
    This model is trained on a dataset (Fake.csv & True.csv).

    ✔ It detects patterns in news text  
    ❌ It does NOT verify real-world truth  

    Accuracy depends on:
    - Input quality
    - Similarity with training data
    """)