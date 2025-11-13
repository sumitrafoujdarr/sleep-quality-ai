# =============================================
# 🌙 AI-Driven Sleep Quality & Recommendation Analyzer
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
        background: rgba(255, 255, 255, 0.85);  /* semi-transparent overlay */
        padding: 2rem;
        border-radius: 1rem;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Put your image file name here
set_background("bg.jpg")  

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="AI Sleep Analyzer", page_icon="🌙", layout="centered")
st.title("🌙 GOODNIGHT (AI-Based Sleep Quality & Recommendation Analyzer)")
st.markdown("Analyze your sleep and get AI-generated personalized recommendations!")

# ----------------------------
# SIMULATED DATASET (500 users)
# ----------------------------
np.random.seed(42)
N = 500
data = {
    'Age': np.random.randint(18, 65, N),
    'Meditation': np.random.choice(['Yes','No'], N),
    'Consistency': np.random.choice(['Yes','No'], N),
    'SleepDuration': np.round(np.random.uniform(4,10,N),1),
    'StressLevel': np.random.randint(0,11,N)
}

df = pd.DataFrame(data)

# ----------------------------
# FUNCTION TO CALCULATE SLEEP SCORE
# ----------------------------
def sleep_score(meditation, consistency, stress, sleep_duration, ideal_min=7, ideal_max=9):
    score = 0
    score += 20 if meditation=="Yes" else 0
    score += 20 if consistency=="Yes" else 0
    # Stress contribution
    score += max(0, 20 - stress*2)  # lower stress, higher score
    # Sleep duration contribution
    if sleep_duration < ideal_min:
        score += (sleep_duration/ideal_min)*20
    elif sleep_duration > ideal_max:
        score += (ideal_max/sleep_duration)*20
    else:
        score += 20
    return round(score,1)

# Generate SleepScore
df['SleepScore'] = df.apply(lambda row: sleep_score(row['Meditation'], row['Consistency'], row['StressLevel'], row['SleepDuration']), axis=1)

# Map SleepScore to SleepQuality
def score_to_quality(score):
    if score < 50:
        return "Poor"
    elif score < 75:
        return "Average"
    else:
        return "Excellent"

df['SleepQuality'] = df['SleepScore'].apply(score_to_quality)

# ----------------------------
# ENCODE CATEGORICAL FEATURES
# ----------------------------
le_meditation = LabelEncoder()
le_consistency = LabelEncoder()
le_quality = LabelEncoder()

df['MeditationEnc'] = le_meditation.fit_transform(df['Meditation'])
df['ConsistencyEnc'] = le_consistency.fit_transform(df['Consistency'])
df['SleepQualityEnc'] = le_quality.fit_transform(df['SleepQuality'])

# ----------------------------
# MODEL: Predict Sleep Quality
# ----------------------------
features = ['MeditationEnc','ConsistencyEnc','StressLevel','SleepDuration']
X = df[features]
y = df['SleepQualityEnc']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_quality = RandomForestClassifier(n_estimators=200, random_state=42)
model_quality.fit(X_train, y_train)

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
# PREDICTION
# ----------------------------
# Encode inputs
med_val = le_meditation.transform([meditation])[0]
con_val = le_consistency.transform([consistency])[0]

input_features = np.array([[med_val, con_val, stress, sleep_duration]])
pred_quality = model_quality.predict(input_features)
pred_quality_label = le_quality.inverse_transform(pred_quality)[0]

# Calculate sleep score
pred_score = sleep_score(meditation, consistency, stress, sleep_duration)

st.markdown("---")
st.success(f"🌙 Predicted Sleep Quality: **{pred_quality_label}**")
st.info(f"🛏 Sleep Duration: {sleep_duration} hours")
st.info(f"🌓 AI Sleep Score: **{pred_score}/100**")

# ----------------------------
# AI-BASED RECOMMENDATIONS
# ----------------------------
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
