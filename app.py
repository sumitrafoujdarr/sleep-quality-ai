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
st.set_page_config(page_title="AI Sleep Quality Analyzer", layout="centered")

# ----------------------------
# HEADER
# ----------------------------
st.title("🌙 AI-Based Sleep Quality Analyzer")
st.markdown("#### Predict your sleep quality & get personalized recommendations")

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv("sleep_dataset.csv")  # make sure the CSV path is correct

# ----------------------------
# ENCODE CATEGORICALS
# ----------------------------
le_med = LabelEncoder()
le_dis = LabelEncoder()
le_quality = LabelEncoder()

df['Meditation'] = le_med.fit_transform(df['Meditation'])
df['Consistency'] = df['Consistency'].map({'Yes':1,'No':0})
df['SleepingDisorder'] = le_dis.fit_transform(df['SleepingDisorder'])
y = le_quality.fit_transform(df['SleepQuality'])

X = df[['Age','SleepDuration','Meditation','Consistency','SleepingDisorder']]

# ----------------------------
# MODEL TRAINING
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
st.markdown(f"📊 **Model Accuracy:** {accuracy_score(y_test, pred)*100:.2f}%")

# ----------------------------
# USER INPUT
# ----------------------------
st.subheader("🧘 Enter your sleep details")

age = st.number_input("Age", min_value=5, max_value=100, step=1)
bedtime = st.time_input("Bedtime", datetime.time(23,0))
wakeuptime = st.time_input("Wakeup Time", datetime.time(7,0))
meditation = st.selectbox("Meditate daily?", ["Yes","No"])
consistency = st.checkbox("Maintain consistent sleep schedule", value=False)
disorder_input = st.selectbox("Disorder symptoms?", ["None","Insomnia","Mild Sleep Disorder","Sleep Apnea"])

# ----------------------------
# CALCULATE SLEEP DURATION
# ----------------------------
bt = datetime.datetime.combine(datetime.date.today(), bedtime)
wt = datetime.datetime.combine(datetime.date.today(), wakeuptime)
if wt <= bt:
    wt += datetime.timedelta(days=1)
sleep_duration = round((wt - bt).total_seconds()/3600,2)
st.info(f"🕒 Estimated Sleep Duration: {sleep_duration} hours")

# Encode user input
med_val = le_med.transform([meditation])[0]
con_val = 1 if consistency else 0
dis_val = le_dis.transform([disorder_input])[0] if disorder_input in le_dis.classes_ else 0

# ----------------------------
# PREDICTION
# ----------------------------
if st.button("✨ Analyze Sleep"):
    input_data = np.array([[age,sleep_duration,med_val,con_val,dis_val]])
    pred_quality = model.predict(input_data)
    sleep_quality = le_quality.inverse_transform(pred_quality)[0]

    st.success(f"🌙 Predicted Sleep Quality: {sleep_quality}")
    st.info(f"🩺 Reported Disorder: {disorder_input}")

    # Age-based ideal sleep duration
    if 6 <= age <= 12:
        ideal = 9
    elif 13 <= age <= 19:
        ideal = 9
    elif 20 <= age <= 35:
        ideal = 8
    elif 36 <= age <= 50:
        ideal = 8
    elif 51 <= age <= 70:
        ideal = 7.5
    else:
        ideal = 7

    st.write(f"🕑 Recommended Sleep Duration for your age: {ideal} hours")

    if sleep_duration < ideal:
        st.warning(f"⚠️ You are sleeping {sleep_duration} hours, below recommended.")
    else:
        st.success("✅ Your sleep duration is sufficient.")

    # Feedback based on predicted quality
    if sleep_quality.lower()=="poor":
        st.error("😴 Your sleep quality seems poor. Follow these tips!")
    elif sleep_quality.lower()=="average":
        st.info("💤 Your sleep is okay — some improvements needed!")
    else:
        st.success("🌟 Excellent sleep quality — Keep it up!")
