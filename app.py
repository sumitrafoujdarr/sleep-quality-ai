import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import datetime
import matplotlib.pyplot as plt
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
# SIMULATED DATASET (500 users)
# ----------------------------
np.random.seed(42)
N = 500
data = {
    'Age': np.random.randint(18, 65, N),
    'Meditation': np.random.choice(['Yes','No'], N),
    'Consistency': np.random.choice(['Yes','No'], N),
    'SleepDuration': np.round(np.random.uniform(4,10,N),1),
    'StressLevel': np.random.randint(0,11,N),
    'Disorder': np.random.choice(['None','Insomnia','Mild'], N)
}

df = pd.DataFrame(data)

# ----------------------------
# GENERATE SLEEP QUALITY
# ----------------------------
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

# ----------------------------
# MODEL 2: Recommendation Category
# ----------------------------
rec_categories = ['Caffeine','Meditation','Exercise','Routine','Stress']
df['RecCategory'] = np.random.choice(rec_categories, N)
le_rec = LabelEncoder()
df['RecCategoryEnc'] = le_rec.fit_transform(df['RecCategory'])

X2 = X1.copy()
y2 = df['RecCategoryEnc']

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)
model_rec = RandomForestClassifier(n_estimators=200, random_state=42)
model_rec.fit(X2_train, y2_train)

# ----------------------------
# FUNCTION TO CHECK SUFFICIENT SLEEP
# ----------------------------
def sleep_enough_by_age(age, duration):
    if 6 <= age <= 12:
        return 9 <= duration <= 11, "9-11 hours"
    elif 13 <= age <= 19:
        return 8 <= duration <= 10, "8-10 hours"
    elif 20 <= age <= 35:
        return 7 <= duration <= 9, "7-9 hours"
    elif 36 <= age <= 50:
        return 7 <= duration <= 9, "7-9 hours"
    elif 51 <= age <= 70:
        return 7 <= duration <= 8, "7-8 hours"
    else:
        return 7 <= duration <= 8, "7-8 hours"

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

# Check sleep sufficiency
enough, ideal_hours = sleep_enough_by_age(age, sleep_duration)
min_hours = int(ideal_hours.split('-')[0])
max_hours = int(ideal_hours.split('-')[1].split()[0])

# Encode user input
med_val = le_meditation.transform([meditation])[0]
con_val = le_consistency.transform([consistency])[0]
dis_val = le_disorder.transform([disorder_input])[0]

input_features = np.array([[age, med_val, con_val, sleep_duration, stress, dis_val]])

# ----------------------------
# PREDICTION & AI RECOMMENDATIONS
# ----------------------------
if st.button("✨ Analyze My Sleep"):
    # Predict sleep quality
    pred_quality = model_quality.predict(input_features)
    quality = le_quality.inverse_transform(pred_quality)[0]

    # Adjust quality based on sleep duration
    if sleep_duration < min_hours:
        quality = "Poor" if sleep_duration < min_hours-1 else "Average"
    elif sleep_duration > max_hours:
        quality = "Average"

    st.markdown("---")
    st.success(f"🌙 Predicted Sleep Quality: **{quality.upper()}**")
    st.info(f"🩺 Disorder: {disorder_input}")
    st.info(f"🛏 Sleep Duration: {sleep_duration} hours")

    # AI Sleep Score
    pred_prob = model_quality.predict_proba(input_features)[0]
    sleep_score = int(pred_prob.max() * 100)
    st.info(f"🌟 AI Sleep Score: **{sleep_score}/100**")

    # Visualize probabilities
    labels = ['Poor', 'Average', 'Excellent']
    sizes = pred_prob
    colors = ['#ff6666','#ffcc66','#66ff66']
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    st.pyplot(fig)

    # AI recommendation mapping
    ai_rec_map = {
        'Caffeine': [
            "AI suggests reducing caffeine intake for better sleep cycles.",
            "AI predicts improved sleep if caffeine is avoided after 3 PM.",
            "Consider herbal teas in the evening for AI-optimized rest."
        ],
        'Meditation': [
            "AI recommends daily meditation to improve sleep quality.",
            "Mindfulness exercises detected as helpful by AI models.",
            "Try 10-25 mins of guided meditation, suggested by AI."
        ],
        'Exercise': [
            "Light evening exercises are predicted to enhance sleep by AI.",
            "AI suggests gentle stretching before bed.",
            "Yoga routines recommended by AI for deeper sleep."
        ],
        'Routine': [
            "Maintain a consistent sleep schedule, per AI analysis.",
            "AI predicts screen-free evenings improve sleep quality.",
            "AI suggests going to bed and waking up at fixed times."
        ],
        'Stress': [
            "AI detects high stress; recommends deep breathing exercises.",
            "AI suggests journaling before bed to reduce stress.",
            "Mindfulness-based stress reduction recommended by AI."
        ]
    }

    # Predict recommendation category
    pred_rec = model_rec.predict(input_features)
    rec_category = le_rec.inverse_transform(pred_rec)[0]
    rec_list = np.random.choice(ai_rec_map[rec_category], np.random.randint(1,4), replace=False).tolist()

    # Add dynamic advice based on user input
    if meditation=="No":
        rec_list.append("Start meditating 10-25 mins daily")
    if consistency=="No":
        rec_list.append("Maintain a consistent sleep schedule")
    if stress>5:
        rec_list.append("Practice stress management techniques")
    if sleep_duration < min_hours:
        rec_list.append(f"Increase sleep to at least {ideal_hours}")
    elif sleep_duration > max_hours:
        rec_list.append(f"Do not oversleep; maintain {ideal_hours}")

    st.markdown("### 💡 AI-Generated Recommendations:")
    for advice in rec_list:
        st.markdown(f"🌝 {advice}")

    # Motivational AI message
    motivation = [
        "AI believes you can improve your sleep tonight 🌙",
        "Sleep is vital! AI recommends following your personalized plan 🛌",
        "Consistency + mindfulness = AI-approved formula for better sleep!"
    ]
    st.markdown(f"💡 {np.random.choice(motivation)}")
