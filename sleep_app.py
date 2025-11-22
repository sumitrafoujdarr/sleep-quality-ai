import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import base64

# ------------------------- BACKGROUND IMAGE ------------------------------
def set_background(local_img_path):
    with open(local_img_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    page_bg_img = f"""
    <style>
    body {{
    background-image: url("data:image/png;base64,{encoded_string}");
    background-size: cover;
    background-attachment: fixed;
    }}
    .stApp {{
        background: rgba(255, 255, 255, 0.85);
        padding: 2rem;
        border-radius: 1rem;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)


# ------------------------- AI ADVICE FUNCTION (ADDED BACK) ------------------------------
def generate_ai_advice(age, stress, duration, quality, disorder):
    return f"""
1. Based on your sleep duration of **{duration} hours** and stress level **{stress}**, try 10 minutes of slow breathing before sleep.
2. Since your predicted sleep quality is **{quality}**, maintain a regular routine and avoid screens 1 hour before bed.
3. Your reported condition: **{disorder}**. Try relaxing activities like meditation or mild stretching.
"""


# ------------------------- MAIN APP ------------------------------
def show_sleep_app():

    st.set_page_config(page_title="AI Sleep Analyzer", page_icon="🌙", layout="centered")
    set_background("bg.jpg")

    st.title("🌙 GOODNIGHT (Fully AI-Based Sleep Quality, Score & Recommendation Analyzer)")

    # Dataset
    github_url = "https://raw.githubusercontent.com/sumitrafoujdarr/sleep-quality-ai/refs/heads/main/sleep_dataset.csv"
    df = pd.read_csv(github_url)
    df["Disorder"] = df["Disorder"].fillna("None")

    # Label Encoders
    le_meditation = LabelEncoder()
    le_consistency = LabelEncoder()
    le_disorder = LabelEncoder()
    le_quality = LabelEncoder()
    le_rec = LabelEncoder()

    df["MeditationEnc"] = le_meditation.fit_transform(df["Meditation"])
    df["ConsistencyEnc"] = le_consistency.fit_transform(df["Consistency"])
    df["DisorderEnc"] = le_disorder.fit_transform(df["Disorder"])
    df["QualityEnc"] = le_quality.fit_transform(df["SleepQuality"])
    df["RecEnc"] = le_rec.fit_transform(df["RecCategory"])

    # ========================= 1. Sleep Quality Model =========================
    X1 = df[["Age", "MeditationEnc", "ConsistencyEnc", "SleepDuration", "StressLevel", "DisorderEnc"]]
    y1 = df["QualityEnc"]

    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
    model_quality = RandomForestClassifier(n_estimators=200, random_state=42)
    model_quality.fit(X1_train, y1_train)

    # ========================= 2. Recommendation Category Model =========================
    X2 = X1.copy()
    y2 = df["RecEnc"]

    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)
    model_rec = RandomForestClassifier(n_estimators=200, random_state=42)
    model_rec.fit(X2_train, y2_train)

    # ========================= 3. Sleep Score Model (Regression) =========================
    score_map = {"Poor": 30, "Average": 60, "Excellent": 90}
    df["SleepScore"] = df["SleepQuality"].map(score_map)

    X3 = X1.copy()
    y3 = df["SleepScore"]

    X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, random_state=42)
    model_score = RandomForestRegressor(n_estimators=200, random_state=42)
    model_score.fit(X3_train, y3_train)

    # ========================= USER INPUTS =========================
    age = st.number_input("Age", 5, 100, 25)
    meditation = st.selectbox("Do you meditate daily?", ["Yes", "No"])
    consistency = st.selectbox("Do you maintain a consistent sleep schedule?", ["Yes", "No"])
    stress = st.slider("Stress Level (0-10)", 0, 10, 5)
    bedtime = st.time_input("Bedtime", datetime.time(23, 0))
    wakeuptime = st.time_input("Wakeup Time", datetime.time(7, 0))

    disorder_options = [d for d in le_disorder.classes_ if str(d).lower() != "nan"]
    disorder_input = st.selectbox("Any disorder symptoms?", disorder_options)

    if disorder_input not in le_disorder.classes_:
        disorder_input = "None"

    # Sleep duration calculation
    bt = datetime.datetime.combine(datetime.date.today(), bedtime)
    wt = datetime.datetime.combine(datetime.date.today(), wakeuptime)
    if wt < bt:
        wt += datetime.timedelta(days=1)

    sleep_duration = round((wt - bt).total_seconds() / 3600, 2)
    st.info(f"🕒 Estimated Sleep Duration: **{sleep_duration} hours**")

    # Encoding input
    med_val = le_meditation.transform([meditation])[0]
    con_val = le_consistency.transform([consistency])[0]
    dis_val = le_disorder.transform([disorder_input])[0]

    input_features = np.array([[age, med_val, con_val, sleep_duration, stress, dis_val]])

    # ========================= PREDICTION =========================
    if st.button("✨ Analyze My Sleep"):

        st.markdown("---")

        # 1. Sleep Quality Prediction
        pred_quality = model_quality.predict(input_features)
        quality = le_quality.inverse_transform(pred_quality)[0]
        st.success(f"🌙 Predicted Sleep Quality: **{quality.upper()}**")

        # 2. Sleep Score Prediction
        predicted_score = int(model_score.predict(input_features)[0])
        predicted_score = max(0, min(100, predicted_score))
        st.info(f"🛏 AI Sleep Score: **{predicted_score}/100**")

        # 3. Recommendation Category Prediction
        pred_rec = model_rec.predict(input_features)
        rec_category = le_rec.inverse_transform(pred_rec)[0]
        st.warning(f"🧭 Recommendation Category: **{rec_category}**")

        # 4. Personalized AI Advice (NOW VISIBLE)
        ai_advice = generate_ai_advice(age, stress, sleep_duration, quality, disorder_input)
        st.markdown("### 💡 AI Personalized Recommendations:")
        st.write(ai_advice)

        # 5. Inspiration Quote
        quotes = [
            "Your future depends on your dreams—so go to sleep.",
            "Sleep is the best meditation.",
            "A well-rested mind is a powerful mind.",
            "Good sleep is the foundation of a healthy life.",
            "Your body heals when you sleep.",
            "Every good day starts the night before.",
            "Rest is not a waste of time; it’s an investment."
        ]
        random_quote = np.random.choice(quotes)
        st.markdown("## 🌟 Sleep Inspiration Quote")
        st.markdown(f"💫 *{random_quote}*")


# Run app
show_sleep_app()
