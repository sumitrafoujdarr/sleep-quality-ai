# =============================================
# 🌙 Fully AI-Based Sleep Quality & Recommendation Analyzer (Context-Aware Recommendations)
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
st.set_page_config(page_title="AI Sleep Analyzer", page_icon="🌙", layout="centered")
st.title("🌙 Fully AI-Driven Sleep Quality & Recommendation Analyzer")
st.markdown("Analyze your sleep and get AI-generated personalized recommendations!")

# ----------------------------
# SIMULATED DATASET
# ----------------------------
np.random.seed(42)
N = 500
data = {
    'Age': np.random.randint(18, 65, N),
    'Meditation': np.random.choice(['Yes','No'], N),
    'Consistency': np.random.choice(['Yes','No'], N),
    'SleepDuration': np.round(np.random.uniform(4,10,N),1),
    'StressLevel': np.random.randint(0,11,N),  # 0-10
    'Disorder': np.random.choice(['None','Insomnia','Mild'], N)
}

df = pd.DataFrame(data)

# ----------------------------
# ASSIGN SLEEP QUALITY BASED ON FEATURES INCLUDING SLEEP DURATION
# ----------------------------
def assign_sleep_quality(row):
    # Recommended sleep hours
    if 6 <= row['Age'] <= 12:
        min_sleep, max_sleep = 9, 11
    elif 13 <= row['Age'] <= 19:
        min_sleep, max_sleep = 8, 10
    elif 20 <= row['Age'] <= 35:
        min_sleep, max_sleep = 7, 9
    elif 36 <= row['Age'] <= 50:
        min_sleep, max_sleep = 7, 9
    elif 51 <= row['Age'] <= 70:
        min_sleep, max_sleep = 7, 8
    else:
        min_sleep, max_sleep = 7, 8

    # Sleep duration impact
    if row['SleepDuration'] < min_sleep - 1:
        return 'Poor'
    elif row['SleepDuration'] < min_sleep:
        return 'Average'

    # Other factors
    score = 0
    score += 1 if row['Meditation']=='Yes' else 0
    score += 1 if row['Consistency']=='Yes' else 0
    score += 1 if row['StressLevel'] <=5 else 0
    score -= 1 if row['Disorder'] != 'None' else 0

    if score <=1: return 'Poor'
    elif score==2: return 'Average'
    else: return 'Excellent'

df['SleepQuality'] = df.apply(assign_sleep_quality, axis=1)

# ----------------------------
# ENCODE FEATURES
# ----------------------------
le_meditation = LabelEncoder()
le_consistency = LabelEncoder()
le_disorder = LabelEncoder()
le_quality = LabelEncoder()

df['Meditation'] = le_meditation.fit_transform(df['Meditation'])
df['Consistency'] = le_consistency.fit_transform(df['Consistency'])
df['Disorder'] = le_disorder.fit_transform(df['Disorder'])
df['SleepQualityEnc'] = le_quality.fit_transform(df['SleepQuality'])

# ----------------------------
# MODEL: Sleep Quality Prediction
# ----------------------------
X = df[['Age','Meditation','Consistency','SleepDuration','StressLevel','Disorder']]
y = df['SleepQualityEnc']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_quality = RandomForestClassifier(n_estimators=200, random_state=42)
model_quality.fit(X_train, y_train)
y_pred = model_quality.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.info(f"🔹 Sleep Quality Model Accuracy: {acc*100:.2f}%")

# ----------------------------
# USER INPUT
# ----------------------------
st.markdown("---")
st.subheader("🧘‍♀️ Enter Your Sleep & Lifestyle Details")

age = st.number_input("Age", 5, 100, 25)
meditation = st.selectbox("Do you meditate daily?", ["Yes","No"])
consistency = st.selectbox("Do you maintain a consistent sleep schedule?", ["Yes","No"])
stress = st.slider("Stress Level (0-10)", 0, 10, 5)
bedtime = st.time_input("Bedtime", datetime.time(23,0))
wakeuptime = st.time_input("Wakeup Time", datetime.time(7,0))
disorder_input = st.selectbox("Any disorder symptoms?", ["None","Insomnia","Mild"])

# Calculate sleep duration
bt = datetime.datetime.combine(datetime.date.today(), bedtime)
wt = datetime.datetime.combine(datetime.date.today(), wakeuptime)
if wt < bt:
    wt += datetime.timedelta(days=1)
sleep_duration = round((wt - bt).total_seconds()/3600,2)
st.info(f"🕒 Estimated Sleep Duration: **{sleep_duration} hours**")

# Encode user input
med_val = le_meditation.transform([meditation])[0]
con_val = le_consistency.transform([consistency])[0]
dis_val = le_disorder.transform([disorder_input])[0]
input_features = np.array([[age, med_val, con_val, sleep_duration, stress, dis_val]])

# ----------------------------
# PREDICTION & CONTEXT-AWARE RECOMMENDATIONS
# ----------------------------
if st.button("✨ Analyze My Sleep"):
    # Predict sleep quality
    pred_quality = model_quality.predict(input_features)
    quality = le_quality.inverse_transform(pred_quality)[0]

    st.markdown("---")
    st.success(f"🌙 Predicted Sleep Quality: **{quality.upper()}**")
    st.info(f"🩺 Disorder: {disorder_input}")
    st.info(f"🛏 Sleep Duration: {sleep_duration} hours")

    # Context-aware AI recommendations
    rec_list = []

    # Check each feature and provide advice if lacking
    if meditation=="No":
        rec_list.append("Meditate 10-25 mins daily to improve sleep quality")
    if consistency=="No":
        rec_list.append("Maintain a consistent sleep schedule")
    if stress>5:
        rec_list.append("Practice stress management: meditation, deep breathing, or light exercise")
    if disorder_input != "None":
        rec_list.append("Follow sleep disorder management advice, consult a specialist if needed")
    # Sleep duration recommendations based on age
    if 20 <= age <= 35:
        ideal = (7,9)
    elif 36 <= age <= 50:
        ideal = (7,9)
    elif 51 <= age <= 70:
        ideal = (7,8)
    else:
        ideal = (7,9)
    if sleep_duration < ideal[0]:
        rec_list.append(f"Increase your sleep to at least {ideal[0]}-{ideal[1]} hours")
    elif sleep_duration > ideal[1]:
        rec_list.append(f"Try not to oversleep, maintain {ideal[0]}-{ideal[1]} hours for optimal health")

    st.markdown("### 💡 AI-Generated Recommendations:")
    for advice in rec_list:
        st.markdown(f"🔹 {advice}")
