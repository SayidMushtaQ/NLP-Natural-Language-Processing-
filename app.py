import streamlit as st
import joblib

# ==========================================
# PAGE SETTINGS
# ==========================================
st.set_page_config(
    page_title="Emotion Classifier",
    page_icon="🧠",
    layout="centered"
)

# ==========================================
# CSS
# ==========================================
st.markdown("""
<style>

.stApp {
    background-color: black;
    color: white;
}

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: white;
}

.subtitle {
    text-align: center;
    color: #aaaaaa;
    margin-bottom: 25px;
}

.result-box {
    background-color: #111111;
    border: 1px solid #333333;
    border-radius: 20px;
    padding: 30px;
    text-align: center;
    margin-top: 20px;
}

.result-text {
    font-size: 38px;
    font-weight: bold;
    color: white;
}

textarea {
    background-color: #111111 !important;
    color: white !important;
    border-radius: 12px !important;
}

div.stButton > button {
    width: 100%;
    height: 50px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    model = joblib.load("svm_model.pkl")
    return vectorizer, model

try:
    tfidf_vectorizer, svc_model = load_model()

except:
    st.error("Model files not found.")
    st.stop()

# ==========================================
# EMOTION MAP
# ==========================================
emotion_map = {
    0: ("😢", "Sadness"),
    1: ("😠", "Anger"),
    2: ("❤️", "Love"),
    3: ("😲", "Surprise"),
    4: ("😨", "Fear"),
    5: ("😊", "Joy")
}

# ==========================================
# HEADER
# ==========================================
st.markdown(
    '<div class="main-title">🧠 Emotion Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">TF-IDF + Linear SVM</div>',
    unsafe_allow_html=True
)

# ==========================================
# INPUT
# ==========================================
text = st.text_area(
    "Enter text",
    height=180,
    placeholder="Type your sentence here..."
)

# ==========================================
# BUTTONS
# ==========================================
col1, col2 = st.columns(2)

with col1:
    predict = st.button("🚀 Predict")

with col2:
    clear = st.button("🗑️ Clear")

if clear:
    st.rerun()

# ==========================================
# PREDICTION
# ==========================================
if predict:

    if text.strip() == "":
        st.warning("Please enter text.")
    else:

        text_tfidf = tfidf_vectorizer.transform([text])

        prediction = svc_model.predict(text_tfidf)[0]

        try:
            prediction = int(prediction)

            emoji, feeling = emotion_map[prediction]

        except:
            feeling = str(prediction).lower()

            text_map = {
                "joy": ("😊", "Joy"),
                "sadness": ("😢", "Sadness"),
                "anger": ("😠", "Anger"),
                "fear": ("😨", "Fear"),
                "love": ("❤️", "Love"),
                "surprise": ("😲", "Surprise")
            }

            emoji, feeling = text_map.get(
                feeling,
                ("🧠", feeling.title())
            )

        st.markdown(
            f"""
            <div class="result-box">
                <h4>Your Feeling</h4>
                <div class="result-text">
                    {emoji} {feeling}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )