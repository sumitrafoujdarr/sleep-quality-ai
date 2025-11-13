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
# LOAD DATA
# ----------------------------
df = pd.read_csv("sleep_dataset.csv")  # Use your CSV file here

# ----------------------------
# ENCODING & MODEL TRAINING
# ----------------------------
le = LabelEncoder()
for col in ['Meditation','Consistency','SleepingDisorder']:
    df[col] = le.fit_transform(df[col])

X = df[['Age','SleepDuration','Meditation','Consistency','SleepingDisorder']]
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
st.markdown(f"📊 **Model Accuracy:** {acc*100:.2f}%")

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
disorder_input = st.selectbox("Do you feel any disorder symptoms?", ["None","Insomnia","Mild Sleep Disorder", "Sleep Apnea"])

# Calculate sleep duration
bt = datetime.datetime.combine(datetime.date.today(), bedtime)
wt = datetime.datetime.combine(datetime.date.today(), wakeuptime)
if wt < bt:
    wt += datetime.timedelta(days=1)
sleep_duration = round((wt - bt).total_seconds()/3600,2)
st.info(f"🕒 Estimated Sleep Duration: **{sleep_duration} hours**")

# Encode user input
med_val = 1 if meditation=="Yes" else 0
con_val = 1 if consistency else 0
dis_val = le.transform([disorder_input])[0] if disorder_input in le.classes_ else 0

# ----------------------------
# PREDICTION
# ----------------------------
if st.button("✨ Analyze My Sleep"):
    input_data = np.array([[age,sleep_duration,med_val,con_val,dis_val]])
    pred_quality = model.predict(input_data)
    sleep_quality = le_quality.inverse_transform(pred_quality)[0]

    # ----------------------------
    # RESULTS
    # ----------------------------
    st.markdown("---")
    st.success(f"🌙 **Predicted Sleep Quality:** {sleep_quality.upper()}")
    st.info(f"🩺 **Reported Disorder:** {disorder_input}")

    # ----------------------------
    # RECOMMENDATION ENGINE
    # ----------------------------
    st.markdown("### 💡 Personalized AI Recommendations")

    # Age-based ideal duration
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

    st.write(f"🕑 **Recommended Sleep Duration for your age:** {ideal} hours")

    # Check if actual sleep is enough
    if sleep_duration < ideal:
        st.warning(f"⚠️ You are sleeping **{sleep_duration} hours**, which is below the recommended amount for your age.")
    else:
        st.success(f"✅ Your sleep duration is sufficient for your age.")

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
        </div>
        """,unsafe_allow_html=True) 
