# =============================================
# 🌙 Fully AI-Based Sleep Quality & Recommendation Analyzer
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
# SIMULATED DATASET (500 users)
# ----------------------------
np.random.seed(42)
N = 500
data = {
    'Age': np.random.randint(18, 65, N),
    'Meditation': np.random.choice(['Yes','No'], N),
    'Consistency': np.random.choice(['Yes','No'], N),
    'SleepDuration': np.round(np.random.uniform(4,10,N),1),
    'StressLevel': np.random.randint(1,11,N),
    'Disorder': np.random.choice(['None','Insomnia','Mild'], N)
}

df = pd.DataFrame(data)

# Generate SleepQuality based on some patterns (for demo)
def generate_quality(row):
    score = 0
    score += 1 if row['Meditation']=='Yes' else 0
    score += 1 if row['Consistency']=='Yes' else 0
    score += 1 if 7<=row['SleepDuration']<=9 else 0
    score += 1 if row['StressLevel']<=5 else 0
    score -= 1 if row['Disorder']!='None' else 0
    if score<=1: return 'Poor'
    elif score==2: return 'Average'
    else: return 'Excellent'

df['SleepQuality'] = df.apply(generate_quality, axis=1)

# Generate AI-based Recommendation category
recommendations = ['Reduce caffeine','Meditate more','Exercise regularly',
                   'Maintain consistent sleep','Manage stress','Keep routine','Hydrate & relax']
# Randomly assign 1-3 recommendations for each row
df['Recommendation'] = df.apply(lambda x: ', '.join(np.random.choice(recommendations, np.random.randint(1,4), replace=False)), axis=1)

# ----------------------------
# ENCODE CATEGORICAL FEATURES
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
# MODEL 1: Sleep Quality Prediction
# ----------------------------
X1 = df[['Age','Meditation','Consistency','SleepDuration','StressLevel','Disorder']]
y1 = df['SleepQualityEnc']

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
model_quality = RandomForestClassifier(n_estimators=200, random_state=42)
model_quality.fit(X1_train, y1_train)

y1_pred = model_quality.predict(X1_test)
acc1 = accuracy_score(y1_test, y1_pred)
st.info(f"🔹 Sleep Quality Model Accuracy: {acc1*100:.2f}%")

# ----------------------------
# MODEL 2: Recommendation Prediction
# ----------------------------
# We’ll encode recommendations as labels
# For multi-label, we can use MultiOutputClassifier, but for simplicity we assign categories
# Here, we convert comma-separated strings into one of 5 categories for demo
rec_categories = ['Caffeine','Meditation','Exercise','Routine','Stress']
df['RecCategory'] = np.random.choice(rec_categories, N)
le_rec = LabelEncoder()
df['RecCategoryEnc'] = le_rec.fit_transform(df['RecCategory'])

X2 = X1.copy()  # same features
y2 = df['RecCategoryEnc']

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)
model_rec = RandomForestClassifier(n_estimators=200, random_state=42)
model_rec.fit(X2_train, y2_train)

y2_pred = model_rec.predict(X2_test)
acc2 = accuracy_score(y2_test, y2_pred)
st.info(f"🔹 Recommendation Model Accuracy: {acc2*100:.2f}%")

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

input_features = np.array([[age, med_val, con_val, sleep_duration, stress, dis_val]])

# ----------------------------
# PREDICTION
# ----------------------------
if st.button("✨ Analyze My Sleep"):
    # Predict sleep quality
    pred_quality = model_quality.predict(input_features)
    quality = le_quality.inverse_transform(pred_quality)[0]
    st.markdown("---")
    st.success(f"🌙 Predicted Sleep Quality: **{quality.upper()}**")
    st.info(f"🩺 Disorder: {disorder_input}")
    st.info(f"🛏 Sleep Duration: {sleep_duration} hours")

    # Predict recommendation category
    pred_rec = model_rec.predict(input_features)
    rec_category = le_rec.inverse_transform(pred_rec)[0]

    # Map category to actual advice (AI-driven)
    rec_map = {
        'Caffeine': ['Reduce caffeine intake', 'Avoid late coffee/tea', 'Drink herbal tea'],
        'Meditation': ['Meditate 10-25 mins daily', 'Practice mindfulness', 'Relaxation exercises'],
        'Exercise': ['Light exercise daily', 'Stretching before bed', 'Yoga for sleep'],
        'Routine': ['Maintain consistent sleep schedule', 'Go to bed & wake up at same time', 'Limit late-night screen use'],
        'Stress': ['Manage stress through meditation', 'Avoid heavy workload before bed', 'Deep breathing exercises']
    }

    st.markdown("### 💡 AI-Generated Recommendations:")
    for advice in rec_map[rec_category]:
        st.markdown(f"🔹 {advice}")
