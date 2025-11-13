# =============================================
# 🌙 AI-Based Sleep Quality & Recommendation Analyzer
# (AI Model: Random Forest Classifier)
# =============================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import datetime
import base64

# ----------------------------
# BACKGROUND IMAGE
# ----------------------------
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

# Set your background image path
set_background("bg.jpg")

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="AI Sleep Analyzer", page_icon="🌙", layout="centered")
st.title("🌙 GOODNIGHT (AI-Based Sleep Quality & Recommendation Analyzer)")
st.markdown("Analyze your sleep and get AI-generated personalized recommendations!")

# ----------------------------
# GENERATE SAMPLE DATA FOR TRAINING
# ----------------------------
np.random.seed(42)
data_size = 500

df = pd.DataFrame({
    "Age": np.random.randint(18, 60, data_size),
    "SleepDuration": np.random.uniform(4, 10, data_size),
    "Consistency": np.random.choice(["Yes","No"], data_size),
    "Stress": np.random.randint(0,11,data_size),
    "Meditation": np.random.choice(["Yes","No"], data_size),
})

# Generate Sleep Quality based on weighted logic (for training)
def generate_quality(row):
    score = 0
    # Sleep Duration
    if row.SleepDuration < 5:
        score += 0
    elif row.SleepDuration < 7:
        score += (row.SleepDuration-5)/2*40
    elif row.SleepDuration <= 9:
        score += 40
    else:
        score += max(0, 40 - (row.SleepDuration-9)*10)
    # Consistency
    score += 30 if row.Consistency=="Yes" else 0
    # Stress
    score += max(0, 20 - row.Stress*2)
    # Meditation
    score += 10 if row.Meditation=="Yes" else 0
    
    if score < 50:
        return "Poor"
    elif score < 75:
        return "Average"
    else:
        return "Excellent"

df["SleepQuality"] = df.apply(generate_quality, axis=1)

# Encode categorical variables
le_consistency = LabelEncoder()
le_meditation = LabelEncoder()
df["Consistency_enc"] = le_consistency.fit_transform(df["Consistency"])
df["Meditation_enc"] = le_meditation.fit_transform(df["Meditation"])

# Features and target
X = df[["Age","SleepDuration","Consistency_enc","Stress","Meditation_enc"]]
y = df["SleepQuality"]

# Train model
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X, y)

# ----------------------------
# USER INPUT
# ----------------------------
age = st.number_input("Age", 5, 100, 25)
meditation = st.selectbox("Do you meditate daily?", ["Yes","No"])
consistency = st.selectbox("Do you maintain a consistent sleep schedule?", ["Yes","No"])
stress = st.slider("Stress Level (0-10)", 0, 10, 5)
bedtime = st.time_input("Bedtime", datetime.time(23,0))
wakeuptime = st.time_input("Wakeup Time", datetime.time(7,0))

# Calculate sleep duration
bt = datetime.datetime.combine(datetime.date.today(), bedtime)
wt = datetime.datetime.combine(datetime.date.today(), wakeuptime)
if wt < bt:
    wt += datetime.timedelta(days=1)
sleep_duration = round((wt - bt).total_seconds()/3600,2)
st.info(f"🕒 Estimated Sleep Duration: **{sleep_duration} hours**")

# ----------------------------
# PREDICTION BUTTON
# ----------------------------
if st.button("✨ Analyze My Sleep"):
    # Encode user input
    consistency_enc = le_consistency.transform([consistency])[0]
    meditation_enc = le_meditation.transform([meditation])[0]

    user_features = np.array([[age, sleep_duration, consistency_enc, stress, meditation_enc]])
    
    # Predict sleep quality
    predicted_quality = rf_model.predict(user_features)[0]
    
    # Predict probability (0-100 AI Sleep Score)
    prob = rf_model.predict_proba(user_features)[0]
    class_idx = list(rf_model.classes_).index(predicted_quality)
    sleep_score = round(prob[class_idx]*100,1)

    st.markdown("---")
    st.success(f"🌙 Predicted Sleep Quality: **{predicted_quality}**")
    st.info(f"🛏 Sleep Duration: {sleep_duration} hours")
    st.info(f"🌓 AI Sleep Score: **{sleep_score}/100**")

    # ----------------------------
    # AI Recommendations
    # ----------------------------
    rec_list = []

    # Sleep Duration priority
    if sleep_duration < 7:
        rec_list.append("Increase sleep duration to at least 7 hours")
    elif sleep_duration > 9:
        rec_list.append("Avoid oversleeping; maintain 7-9 hours of sleep")

    # Consistency
    if consistency=="No":
        rec_list.append("Maintain a consistent sleep schedule")

    # Stress
    if stress > 5:
        rec_list.append("Practice stress management techniques (meditation, deep breathing)")

    # Meditation
    if meditation=="No":
        rec_list.append("Start meditating 10-25 mins daily")

    st.markdown("### 💡 AI-Generated Recommendations:")
    for advice in rec_list:
        st.markdown(f"🌝 {advice}")
