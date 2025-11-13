import streamlit as st
import pandas as pd
import numpy as np
import datetime
import base64
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

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
# SIMULATED DATASET
# ----------------------------
np.random.seed(42)
N = 1000
data = {
    'Age': np.random.randint(18, 65, N),
    'Meditation': np.random.choice(['Yes','No'], N),
    'Consistency': np.random.choice(['Yes','No'], N),
    'SleepDuration': np.round(np.random.uniform(4,10,N),1),
    'StressLevel': np.random.randint(0,11,N),
    'Disorder': np.random.choice(['None','Insomnia','Mild'], N)
}
df = pd.DataFrame(data)

# Generate Sleep Quality (label)
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

# Encode categorical features
le_meditation = LabelEncoder()
le_consistency = LabelEncoder()
le_disorder = LabelEncoder()
le_quality = LabelEncoder()

df['MeditationEnc'] = le_meditation.fit_transform(df['Meditation'])
df['ConsistencyEnc'] = le_consistency.fit_transform(df['Consistency'])
df['DisorderEnc'] = le_disorder.fit_transform(df['Disorder'])
df['SleepQualityEnc'] = le_quality.fit_transform(df['SleepQuality'])

# Features and target
features = ['Age','MeditationEnc','ConsistencyEnc','SleepDuration','StressLevel','DisorderEnc']
X = df[features]
y_quality = df['SleepQualityEnc']

# ----------------------------
# Train Sleep Quality Model
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y_quality, test_size=0.2, random_state=42)
model_quality = RandomForestClassifier(n_estimators=300, random_state=42)
model_quality.fit(X_train, y_train)

# ----------------------------
# Train Sleep Score Model (0-100)
# ----------------------------
df['SleepScore'] = df.apply(lambda row: sleep_score(
    row['Age'], 
    "Yes" if row['Meditation']==1 else "No", 
    "Yes" if row['Consistency']==1 else "No", 
    row['StressLevel'], 
    row['SleepDuration']
), axis=1)

# Features
X = df[['Age','Meditation','Consistency','SleepDuration','StressLevel']]
y = df['SleepScore']

# Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_score.fit(X_train, y_score)

# ----------------------------
# Train Recommendation Model
# ----------------------------
rec_categories = ['Caffeine','Meditation','Exercise','Routine','Stress']
df['RecCategory'] = np.random.choice(rec_categories, N)
le_rec = LabelEncoder()
df['RecCategoryEnc'] = le_rec.fit_transform(df['RecCategory'])
y_rec = df['RecCategoryEnc']

model_rec = RandomForestClassifier(n_estimators=300, random_state=42)
model_rec.fit(X_train, y_rec)

# ----------------------------
# USER INPUT
# ----------------------------
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

# Encode inputs
med_val = le_meditation.transform([meditation])[0]
con_val = le_consistency.transform([consistency])[0]
dis_val = le_disorder.transform([disorder_input])[0]
input_features = np.array([[age, med_val, con_val, sleep_duration, stress, dis_val]])

# ----------------------------
# AI-Driven Predictions
# ----------------------------
if st.button("✨ Analyze My Sleep"):
    # Sleep Quality
    pred_quality = model_quality.predict(input_features)
    quality = le_quality.inverse_transform(pred_quality)[0]
    
    # Sleep Score
    pred_score = model_score.predict(input_features)[0]
    
    # Recommendation Category
    pred_rec = model_rec.predict(input_features)
    rec_category = le_rec.inverse_transform(pred_rec)[0]
    
    # Recommendation Map
    rec_map = {
        'Caffeine': ['Reduce caffeine intake', 'Avoid late coffee/tea', 'Drink herbal tea'],
        'Meditation': ['Meditate 10-25 mins daily', 'Practice mindfulness', 'Relaxation exercises'],
        'Exercise': ['Light exercise daily', 'Stretching before bed', 'Yoga for sleep'],
        'Routine': ['Maintain consistent sleep schedule', 'Limit late-night screen use'],
        'Stress': ['Manage stress through meditation', 'Avoid heavy workload before bed', 'Deep breathing exercises']
    }
    
    rec_list = rec_map[rec_category]
    
    st.markdown("---")
    st.success(f"🌙 Predicted Sleep Quality: **{quality.upper()}**")
    st.info(f"🧮 AI Sleep Score: **{pred_score:.1f}/100**")
    st.info(f"🛏 Sleep Duration: {sleep_duration} hours")
    st.info(f"🩺 Disorder: {disorder_input}")
    
    st.markdown("### 💡 AI-Generated Recommendations:")
    for advice in rec_list:
        st.markdown(f"🌝 {advice}")
