# =============================================
# 🌙 AI-Driven Sleep Quality & Recommendation Analyzer
# =============================================

import streamlit as st
import numpy as np
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

set_background("bg.jpg")  

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="AI Sleep Analyzer", page_icon="🌙", layout="centered")
st.title("🌙 GOODNIGHT (AI-Based Sleep Quality & Recommendation Analyzer)")
st.markdown("Analyze your sleep and get AI-generated personalized recommendations!")

# ----------------------------
# FUNCTION TO CALCULATE SLEEP SCORE
# ----------------------------
def sleep_score(meditation, consistency, stress, sleep_duration):
    score = 0
    score += 25 if meditation=="Yes" else 0
    score += 25 if consistency=="Yes" else 0
    score += max(0, 25 - stress*2)  # Lower stress = higher score
    # Sleep duration contribution
    if sleep_duration < 7:
        score += max(0, (sleep_duration/7)*25)
    elif sleep_duration > 9:
        score += max(0, (9/sleep_duration)*25)
    else:
        score += 25
    return round(score,1)

def score_to_quality(score, sleep_duration, stress):
    # Rule-based constraints
    if sleep_duration < 5 or stress >= 9:
        return "Poor"
    elif score < 50:
        return "Poor"
    elif score < 75:
        return "Average"
    else:
        return "Excellent"

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
# ANALYZE BUTTON
# ----------------------------
if st.button("✨ Analyze My Sleep"):

    # Calculate AI Sleep Score
    score = sleep_score(meditation, consistency, stress, sleep_duration)

    # Determine Sleep Quality
    quality = score_to_quality(score, sleep_duration, stress)

    st.markdown("---")
    st.success(f"🌙 Predicted Sleep Quality: **{quality}**")
    st.info(f"🛏 Sleep Duration: {sleep_duration} hours")
    st.info(f"🌓 AI Sleep Score: **{score}/100**")

    # AI-based recommendations
    rec_list = []
    if meditation=="No":
        rec_list.append("Start meditating 10-25 mins daily")
    if consistency=="No":
        rec_list.append("Maintain a consistent sleep schedule")
    if stress>5:
        rec_list.append("Practice stress management techniques")
    if sleep_duration < 7:
        rec_list.append("Increase sleep duration to at least 7 hours")
    elif sleep_duration > 9:
        rec_list.append("Avoid oversleeping; maintain 7-9 hours of sleep")

    st.markdown("### 💡 AI-Generated Recommendations:")
    for advice in rec_list:
        st.markdown(f"🌝 {advice}")
