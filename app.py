# =============================================
# 🌙 AI-Based Sleep Quality & Disorder Analyzer
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
# EMBEDDED DATASET (EXAMPLE)
# ----------------------------
data = {
    'Age': [20,22,25,30,28,21,24,26,29,31], 'Meditation': ['Yes','No','Yes','No','Yes','No','Yes','No','Yes','No'], 'Consistency': ['Yes','No','Yes','No','Yes','Yes','Yes','No','Yes','No'], 'SleepDuration': [8.0,6.5,8.0,6.0,8.0,7.0,7.5,6.0,8.5,6.5], 'SleepQuality': ['Excellent','Poor','Excellent','Poor','Excellent','Average','Excellent','Poor','Excellent','Poor'], 'SleepingDisorder': ['None','Insomnia','None','Insomnia','None','None','None','Insomnia','None','Insomnia'] }



# ----------------------------
# CALCULATE SUFFICIENT SLEEP BASED ON AGE
# ----------------------------
def is_sleep_enough(age, duration):
    """Return 1 if sleep duration is within recommended hours, else 0"""
    if 6 <= age <= 12:
        return 1 if 9 <= duration <= 11 else 0
    elif 13 <= age <= 19:
        return 1 if 8 <= duration <= 10 else 0
    elif 20 <= age <= 35:
        return 1 if 7 <= duration <= 9 else 0
    elif 36 <= age <= 50:
        return 1 if 7 <= duration <= 9 else 0
    elif 51 <= age <= 70:
        return 1 if 7 <= duration <= 8 else 0
    else:
        return 1 if 7 <= duration <= 8 else 0

df['SleepEnough'] = df.apply(lambda row: is_sleep_enough(row['Age'], row['SleepDuration']), axis=1)

# ----------------------------
# ENCODING & MODEL TRAINING
# ----------------------------
le = LabelEncoder()
for col in ['Meditation','Consistency','SleepingDisorder']:
    df[col] = le.fit_transform(df[col])

X = df[['Age','SleepDuration','Meditation','Consistency','SleepingDisorder','SleepEnough']]
y = df['SleepQuality']
le_quality = LabelEncoder()
y_encoded = le_quality.fit_transform(y)

# Train-test split & model
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# Evaluate model accuracy
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)


# ----------------------------
# USER INPUT
# ----------------------------
st.markdown("---")
st.subheader("🧘‍♀️ Enter Your Sleep Details")

age = st.number_input("Age", min_value=5, max_value=100, step=1)
bedtime = st.time_input("Bedtime", datetime.time(23,0))
wakeuptime = st.time_input("Wakeup Time", datetime.time(7,0))
meditation = st.selectbox("Do you meditate daily?", ["Yes","No"])
consistency = st.checkbox("I maintain a consistent sleep schedule", value=False)
disorder_input = st.selectbox("Do you feel any disorder symptoms?", ["None","Insomnia","Mild Sleep Disorder"])

# Calculate sleep duration
bt = datetime.datetime.combine(datetime.date.today(), bedtime)
wt = datetime.datetime.combine(datetime.date.today(), wakeuptime)
if wt < bt:
    wt += datetime.timedelta(days=1)
sleep_duration = round((wt - bt).total_seconds()/3600,2)
st.info(f"🕒 Estimated Sleep Duration: **{sleep_duration} hours**")

# Check if sleep is enough
sleep_enough = is_sleep_enough(age, sleep_duration)

# Encode user input
med_val = 1 if meditation=="Yes" else 0
con_val = 1 if consistency else 0
dis_val = le.transform([disorder_input])[0] if disorder_input in le.classes_ else 0

# ----------------------------
# PREDICTION
# ----------------------------
if st.button("✨ Analyze My Sleep"):
    input_data = np.array([[age,sleep_duration,med_val,con_val,dis_val,sleep_enough]])
    pred_quality = model.predict(input_data)
    sleep_quality = le_quality.inverse_transform(pred_quality)[0]

    # ----------------------------
    # RESULTS
    # ----------------------------
    st.markdown("---")
    st.success(f"🌙 **Predicted Sleep Quality:** {sleep_quality.upper()}")
    st.info(f"🩺 **Reported Disorder:** {disorder_input}")
    st.info(f"🛏 **Sleep Duration Status:** {'Sufficient' if sleep_enough==1 else 'Insufficient'}")

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
    if sleep_quality.lower()=="poor":
        st.error("😴 Your sleep quality seems poor. Follow recommendations!")
        st.markdown("""
        <div class='recommendation'>
        🔹 Avoid screens 30 mins before sleep  
        🔹 No caffeine after evening  
        🔹 Meditate daily 15–25 mins  
        🔹 Light exercise regularly  
        🔹 Avoid stress before bed  
        🔹 Ensure you get enough sleep hours for your age
        </div>
        """,unsafe_allow_html=True)
    elif sleep_quality.lower()=="average":
        st.info("💤 Your sleep is okay — small improvements needed!")
        st.markdown("""
        <div class='recommendation'>
        🔹 Keep consistent bedtime  
        🔹 Reduce late-night screen exposure  
        🔹 Drink water before bed, avoid heavy meals  
        🔹 Meditate 10–15 mins  
        🔹 Check if your sleep duration meets age recommendations
        </div>
        """,unsafe_allow_html=True)
    else:
        st.success("🌟 Excellent Sleep Quality — Keep it up!")
        st.markdown("""
        <div class='recommendation'>
        🔹 Maintain healthy routine  
        🔹 Avoid overworking late nights  
        🔹 Stay hydrated and stress-free  
        🔹 Continue mindfulness & balance  
        🔹 Your sleep duration is sufficient — keep it consistent!
        </div>
        """,unsafe_allow_html=True)
