# =============================================
# 🌙 Fully AI-Based Sleep Quality Analyzer
# =============================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import datetime

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="AI Sleep Quality Analyzer", page_icon="🌙", layout="centered")

st.title("🌙 Fully AI-Driven Sleep Quality Analyzer")
st.markdown("#### Your Smart AI Sleep Coach — Analyze & Improve!")

# ----------------------------
# SIMULATED DATASET (for demo)
# ----------------------------
# 500 users with age, meditation, consistency, sleep duration, stress, disorder, quality
np.random.seed(42)
N = 500
data = {
    'Age': np.random.randint(18, 65, N),
    'Meditation': np.random.choice(['Yes','No'], N),
    'Consistency': np.random.choice(['Yes','No'], N),
    'SleepDuration': np.round(np.random.uniform(4,10,N),1),
    'StressLevel': np.random.randint(1,11,N),
    'Disorder': np.random.choice(['None','Insomnia','Mild'], N),
    'SleepQuality': np.random.choice(['Poor','Average','Excellent'], N, p=[0.3,0.4,0.3])
}
df = pd.DataFrame(data)

# ----------------------------
# PREPROCESS DATA
# ----------------------------
le_meditation = LabelEncoder()
le_consistency = LabelEncoder()
le_disorder = LabelEncoder()
le_quality = LabelEncoder()

df['Meditation'] = le_meditation.fit_transform(df['Meditation'])
df['Consistency'] = le_consistency.fit_transform(df['Consistency'])
df['Disorder'] = le_disorder.fit_transform(df['Disorder'])
df['SleepQualityEncoded'] = le_quality.fit_transform(df['SleepQuality'])

# Features & Target
X = df[['Age','Meditation','Consistency','SleepDuration','StressLevel','Disorder']]
y = df['SleepQualityEncoded']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate (optional)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.info(f"🔹 Model Accuracy on Test Data: {acc*100:.2f}%")

# ----------------------------
# USER INPUT
# ----------------------------
st.markdown("---")
st.subheader("🧘‍♀️ Enter Your Sleep & Lifestyle Details")

age = st.number_input("Age", 5, 100, 25)
meditation = st.selectbox("Do you meditate daily?", ["Yes","No"])
consistency = st.selectbox("Do you maintain a consistent sleep schedule?", ["Yes","No"])
stress = st.slider("Stress Level (1-10)", 1, 10, 5)
bedtime = st.time_input("Bedtime", datetime.time(23,0))
wakeuptime = st.time_input("Wakeup Time", datetime.time(7,0))
disorder_input = st.selectbox("Any disorder symptoms?", ["None","Insomnia","Mild"])

# Calculate sleep duration
bt = datetime.datetime.combine(datetime.date.today(), bedtime)
wt = datetime.datetime.combine(datetime.date.today(), wakeuptime)
if wt < bt:
    wt += datetime.timedelta(days=1)
sleep_duration = round((wt - bt).total_seconds()/3600,2)
st.info(f"🕒 Estimated Sleep Duration: {sleep_duration} hours")

# Encode user input
med_val = le_meditation.transform([meditation])[0]
con_val = le_consistency.transform([consistency])[0]
dis_val = le_disorder.transform([disorder_input])[0]

# ----------------------------
# PREDICTION
# ----------------------------
if st.button("✨ Analyze My Sleep"):
    input_features = np.array([[age,med_val,con_val,sleep_duration,stress,dis_val]])
    pred = model.predict(input_features)
    quality = le_quality.inverse_transform(pred)[0]

    st.markdown("---")
    st.success(f"🌙 Predicted Sleep Quality: **{quality.upper()}**")
    st.info(f"🩺 Disorder: {disorder_input}")
    st.info(f"🛏 Sleep Duration: {sleep_duration} hours")

    # AI-driven Recommendations (simple ML-based mapping)
    if quality=="Poor":
        advice = [
            "🔹 Avoid screens 30 mins before sleep",
            "🔹 Reduce caffeine intake",
            "🔹 Meditate 15-25 mins daily",
            "🔹 Light exercise regularly",
            "🔹 Maintain consistent sleep schedule",
            "🔹 Reduce stress levels"
        ]
    elif quality=="Average":
        advice = [
            "🔹 Keep consistent bedtime",
            "🔹 Meditate 10-15 mins",
            "🔹 Avoid heavy meals before bed",
            "🔹 Light physical activity daily",
            "🔹 Try to sleep at least recommended hours"
        ]
    else:
        advice = [
            "🔹 Maintain healthy routine",
            "🔹 Continue mindfulness and meditation",
            "🔹 Stay hydrated and stress-free",
            "🔹 Keep your sleep schedule consistent"
        ]
    
    for item in advice:
        st.markdown(item)
