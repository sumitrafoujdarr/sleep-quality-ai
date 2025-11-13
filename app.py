# =============================================
# 🌙 AI-Based Sleep Quality & Disorder Analyzer
# =============================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import datetime

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="AI Sleep Quality & Disorder Analyzer", page_icon="🌙", layout="centered")

# ----------------------------
# CUSTOM CSS
# ----------------------------
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #cfd9df 0%, #e2ebf0 100%);
        color: #0b3d91;
        font-family: 'Poppins', sans-serif;
    }
    .stButton>button {
        background-color: #5b8dee;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        height: 50px;
        width: 100%;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #3b6ed0;
    }
    h1, h2, h3, h4 {
        text-align: center;
        color: #0b3d91;
    }
    .recommendation {
        background-color: #f0f9ff;
        border-left: 6px solid #5b8dee;
        padding: 10px;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# HEADER
# ----------------------------
st.title("🌙 AI-Based Sleep Quality & Disorder Analyzer")
st.markdown("#### Your Smart AI Sleep Coach — Analyze, Improve & Rise Refreshed! 🧠💤")

# ----------------------------
# LOAD DATASET
# ----------------------------
uploaded_file = st.file_uploader("📂 Upload your sleep_data.csv file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset successfully loaded!")

    # ----------------------------
    # ENCODING & MODEL TRAINING
    # ----------------------------
    le = LabelEncoder()
    for col in ['Meditation', 'Consistency', 'SleepingDisorder']:
        df[col] = le.fit_transform(df[col])

    # Separate features & targets
    X = df[['Age', 'Meditation', 'Consistency', 'SleepDuration']]
    y_quality = df['SleepQuality']
    y_disorder = df['SleepingDisorder']

    # Encode targets
    le_quality = LabelEncoder()
    le_disorder = LabelEncoder()
    yq = le_quality.fit_transform(y_quality)
    yd = le_disorder.fit_transform(y_disorder)

    # Train-test split for AI model
    X_train, X_test, yq_train, yq_test = train_test_split(X, yq, test_size=0.2, random_state=42)
    model_quality = RandomForestClassifier(n_estimators=150, random_state=42)
    model_quality.fit(X_train, yq_train)

    # Evaluate model accuracy
    pred_q = model_quality.predict(X_test)
    acc_quality = accuracy_score(yq_test, pred_q)

    # Train disorder model
    X_train, X_test, yd_train, yd_test = train_test_split(X, yd, test_size=0.2, random_state=42)
    model_disorder = RandomForestClassifier(n_estimators=150, random_state=42)
    model_disorder.fit(X_train, yd_train)
    pred_d = model_disorder.predict(X_test)
    acc_disorder = accuracy_score(yd_test, pred_d)

    st.markdown(f"📊 **Model Accuracy:** Sleep Quality = {acc_quality*100:.2f}% | Disorder = {acc_disorder*100:.2f}%")

    # ----------------------------
    # USER INPUT SECTION
    # ----------------------------
    st.markdown("---")
    st.subheader("🧘‍♀️ Enter Your Sleep Details")

    age = st.number_input("Age", min_value=5, max_value=100, step=1)
    bedtime = st.time_input("Bedtime", datetime.time(23, 0))
    wakeuptime = st.time_input("Wakeup Time", datetime.time(7, 0))
    meditation = st.selectbox("Do you meditate daily?", ["Yes", "No"])
    consistency = st.checkbox("I maintain a consistent sleep schedule", value=False)
    disorder_input = st.selectbox("Do you feel any disorder symptoms?", ["None", "Mild Sleep Disorder", "Insomnia"])

    # Calculate duration
    bt = datetime.datetime.combine(datetime.date.today(), bedtime)
    wt = datetime.datetime.combine(datetime.date.today(), wakeuptime)
    if wt < bt:
        wt += datetime.timedelta(days=1)
    sleep_duration = round((wt - bt).total_seconds() / 3600, 2)
    st.info(f"🕒 Estimated Sleep Duration: **{sleep_duration} hours**")

    # Encode user input
    med_val = 1 if meditation == "Yes" else 0
    con_val = 1 if consistency else 0

    # ----------------------------
    # AI PREDICTION
    # ----------------------------
    if st.button("✨ Analyze My Sleep"):
        input_data = np.array([[age, med_val, con_val, sleep_duration]])
        quality_pred = model_quality.predict(input_data)
        disorder_pred = model_disorder.predict(input_data)

        sleep_quality = le_quality.inverse_transform(quality_pred)[0]
        sleep_disorder = le_disorder.inverse_transform(disorder_pred)[0]

        # If user selected disorder manually
        if disorder_input != "None":
            sleep_disorder = disorder_input

        # ----------------------------
        # RESULTS
        # ----------------------------
        st.markdown("---")
        st.success(f"🌙 **Predicted Sleep Quality:** {sleep_quality.upper()}")
        st.info(f"🩺 **Detected/Reported Disorder:** {sleep_disorder}")

        # ----------------------------
        # RECOMMENDATION ENGINE
        # ----------------------------
        st.markdown("### 💡 Personalized AI Recommendations")

        # Age-based ideal duration
        if 6 <= age <= 12:
            ideal = "9–11 hours"
        elif 13 <= age <= 19:
            ideal = "8–10 hours"
        elif 20 <= age <= 35:
            ideal = "7–9 hours"
        elif 36 <= age <= 50:
            ideal = "7–9 hours"
        elif 51 <= age <= 70:
            ideal = "7–8 hours"
        else:
            ideal = "7–8 hours"

        st.write(f"🕑 **Recommended Sleep Duration:** {ideal}")

        # Sleep quality-based feedback
        st.markdown("---")

        if sleep_quality.lower() == "poor":
            st.error("😴 Your sleep quality seems poor. You need some self-care & discipline!")
            st.markdown("""
            <div class='recommendation'>
            🔹 Avoid **screens** at least 30 minutes before sleeping  
            🔹 No **caffeine or tea** after evening  
            🔹 Practice **25 minutes of meditation** daily  
            🔹 Include **light exercise** in your routine  
            🔹 Avoid **stressful talks or work** before bed  
            🔹 Try relaxing music or breathing exercises  
            </div>
            """, unsafe_allow_html=True)
            st.warning("✨ *Your body needs care. Consistency heals everything — start small!*")

        elif sleep_quality.lower() == "average":
            st.info("💤 You’re sleeping fairly well — just a few tweaks can make it perfect!")
            st.markdown("""
            <div class='recommendation'>
            🔹 Maintain a **consistent bedtime**  
            🔹 Reduce **late-night screen exposure**  
            🔹 Drink **water before bed** but avoid heavy meals  
            🔹 **Meditation** for 10–15 minutes  
            </div>
            """, unsafe_allow_html=True)
            st.info("🌟 *You’re improving — stay consistent, peace follows!*")

        else:
            st.success("🌟 Excellent Sleep Quality — You’re doing amazing!")
            st.markdown("""
            <div class='recommendation'>
            🔹 Keep up your **healthy routine**  
            🔹 Avoid overworking late nights  
            🔹 Stay **hydrated** and stress-free  
            🔹 Continue **mindfulness and balance**  
            </div>
            """, unsafe_allow_html=True)
            st.success("✨ *Great habits create great energy — keep shining!*")

else:
    st.info("⬆️ Please upload your dataset (`sleep_data.csv`) to start the AI analysis.")
