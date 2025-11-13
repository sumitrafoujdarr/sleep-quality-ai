import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import datetime

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="AI Sleep Quality & Disorder Analyzer", page_icon="🌙", layout="centered")

# ----------------------------
# CUSTOM STYLING
# ----------------------------
st.markdown("""
    <style>
    .main { background-color: #f2f7fa; }
    .stButton>button {
        background-color: #4a90e2;
        color: white;
        font-size: 17px;
        border-radius: 12px;
        height: 50px;
        width: 100%;
    }
    h1, h2, h3 {
        text-align: center;
        color: #1f3c73;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# HEADER
# ----------------------------
st.title("🌙 AI-Based Sleep Quality & Disorder Analyzer")
st.markdown("#### Get AI-powered insights about your sleep habits and mental wellness 🧠💤")

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv("sleep_data.csv")

# Encode categorical columns
encoder = LabelEncoder()
for col in ['Meditation', 'Consistency', 'SleepingDisorder']:
    df[col] = encoder.fit_transform(df[col])

# Define features and targets
X = df[['Age', 'Meditation', 'Consistency', 'SleepDuration', 'SleepingDisorder']]
y = df['SleepQuality']
le_quality = LabelEncoder()
y_encoded = le_quality.fit_transform(y)

# Train model
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X, y_encoded)

# ----------------------------
# USER INPUTS
# ----------------------------
st.subheader("🧘‍♀️ Enter Your Sleep Details")

age = st.number_input("Age", min_value=5, max_value=100, step=1)
bedtime = st.time_input("Bedtime (24-hour format)", datetime.time(23, 0))
wakeuptime = st.time_input("Wakeup Time (24-hour format)", datetime.time(7, 0))
meditation = st.selectbox("Do you meditate daily?", ["Yes", "No"])
consistency = st.checkbox("I maintain a consistent sleep schedule", value=False)
disorder = st.selectbox("Do you have any sleep disorder?", ["None", "Mild Sleep Disorder", "Insomnia"])

# ----------------------------
# CALCULATE SLEEP DURATION
# ----------------------------
bt = datetime.datetime.combine(datetime.date.today(), bedtime)
wt = datetime.datetime.combine(datetime.date.today(), wakeuptime)
if wt < bt:
    wt += datetime.timedelta(days=1)
sleep_duration = round((wt - bt).total_seconds() / 3600, 2)
st.write(f"🕒 Estimated Sleep Duration: **{sleep_duration} hours**")

# ----------------------------
# ENCODE INPUT
# ----------------------------
med_val = 1 if meditation == "Yes" else 0
con_val = 1 if consistency else 0
disorder_map = {"None": 0, "Mild Sleep Disorder": 1, "Insomnia": 2}
disorder_val = disorder_map[disorder]

# ----------------------------
# AI PREDICTION
# ----------------------------
if st.button("✨ Analyze My Sleep"):
    input_data = np.array([[age, med_val, con_val, sleep_duration, disorder_val]])
    prediction = model.predict(input_data)
    sleep_quality = le_quality.inverse_transform(prediction)[0]

    # ----------------------------
    # DISPLAY RESULTS
    # ----------------------------
    st.success(f"🌙 Predicted Sleep Quality: **{sleep_quality.upper()}**")
    st.info(f"🩺 Current Disorder Status: **{disorder}**")

    # ----------------------------
    # AI RECOMMENDATIONS
    # ----------------------------
    st.markdown("---")
    st.subheader("💡 AI-Based Personalized Recommendations")

    # Recommended Sleep Duration by Age
    if 6 <= age <= 12:
        needed = "9–11 hours"
    elif 13 <= age <= 19:
        needed = "8–10 hours"
    elif 20 <= age <= 35:
        needed = "7–9 hours"
    elif 36 <= age <= 50:
        needed = "7–9 hours"
    elif 51 <= age <= 70:
        needed = "7–8 hours"
    else:
        needed = "7–8 hours"

    st.write(f"🕑 **Recommended Sleep Duration:** {needed}")

    # ----------------------------
    # DETAILED RECOMMENDATIONS BASED ON QUALITY
    # ----------------------------
    if sleep_quality.lower() == "poor":
        st.warning("😴 Your sleep quality is **Poor** — You need to take care of your mind and body!")
        st.markdown("""
        **Follow these tips for improvement:**
        - Maintain a **consistent sleep schedule**
        - 📱 Avoid **screen time 30–60 minutes before bed**
        - ☕ Avoid **caffeine or tea after 6 PM**
        - 🧘‍♀️ Practice **25 minutes of meditation** before sleeping
        - 🏃‍♂️ Engage in **light physical exercise** daily
        - 💭 Avoid stress or heavy thoughts before bed
        - 🌿 Keep your sleeping environment calm and dark
        """)
        st.info("✨ *Small consistent habits today will create peaceful nights tomorrow!*")

    elif sleep_quality.lower() == "average":
        st.info("😌 Your sleep quality is **Average** — You’re on the right path, just need some balance!")
        st.markdown("""
        **Try to make it better:**
        - Go to bed **30 minutes earlier**
        - 🌙 Reduce blue light and phone use before sleep
        - 💆 Practice short meditation or deep breathing
        - ☀️ Get sunlight exposure in the morning
        - 🧘‍♂️ Exercise or walk daily for 30 minutes
        """)
        st.success("💫 *Your mind is learning to rest better — keep improving!*")

    else:
        st.success("🌟 Your sleep quality is **Excellent** — Keep it up!")
        st.markdown("""
        **To maintain it:**
        - Continue **consistent bedtime routine**
        - Keep avoiding **screen and caffeine before sleep**
        - 🧘‍♀️ Stay mindful through meditation
        - 💪 Maintain physical and mental fitness
        """)
        st.balloons()
        st.info("🌸 *A peaceful night leads to a powerful day!*")

