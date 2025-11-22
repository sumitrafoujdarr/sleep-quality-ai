import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import base64

# ------------------------- BACKGROUND IMAGE ------------------------------
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

# ------------------------- PAGE CONFIG ------------------------------
st.set_page_config(page_title="AI Sleep Analyzer", page_icon="🌙", layout="centered")
st.title("🌙 GOODNIGHT (AI-Based Sleep Quality & Recommendation Analyzer)")

# ------------------------- LOAD REAL DATASET ------------------------------
github_url = "https://raw.githubusercontent.com/sumitrafoujdarr/sleep-quality-ai/refs/heads/main/sleep_dataset.csv"

df = pd.read_csv(github_url)

# FIX: Replace NaN in Disorder with "None"
df["Disorder"] = df["Disorder"].fillna("None")

# --------------------- ENCODING LABELS -------------------
le_meditation = LabelEncoder()
le_consistency = LabelEncoder()
le_disorder = LabelEncoder()
le_quality = LabelEncoder()
le_rec = LabelEncoder()

df["MeditationEnc"] = le_meditation.fit_transform(df["Meditation"])
df["ConsistencyEnc"] = le_consistency.fit_transform(df["Consistency"])
df["DisorderEnc"] = le_disorder.fit_transform(df["Disorder"])
df["QualityEnc"] = le_quality.fit_transform(df["SleepQuality"])
df["RecEnc"] = le_rec.fit_transform(df["RecCategory"])

# --------------------- MODEL 1 — Sleep Quality Prediction ---------------------
X1 = df[["Age", "MeditationEnc", "ConsistencyEnc", "SleepDuration", "StressLevel", "DisorderEnc"]]
y1 = df["QualityEnc"]

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
model_quality = RandomForestClassifier(n_estimators=200, random_state=42)
model_quality.fit(X1_train, y1_train)

# --------------------- MODEL 2 — Recommendation Category Prediction ---------------------
X2 = X1.copy()
y2 = df["RecEnc"]

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)
model_rec = RandomForestClassifier(n_estimators=200, random_state=42)
model_rec.fit(X2_train, y2_train)

# ------------------------------ SLEEP RANGE CHECK ------------------------------
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

# ------------------------------ USER INPUTS ------------------------------
age = st.number_input("Age", 5, 100, 25)
meditation = st.selectbox("Do you meditate daily?", ["Yes", "No"])
consistency = st.selectbox("Do you maintain a consistent sleep schedule?", ["Yes", "No"])
stress = st.slider("Stress Level (0-10)", 0, 10, 5)
bedtime = st.time_input("Bedtime", datetime.time(23,0))
wakeuptime = st.time_input("Wakeup Time", datetime.time(7,0))

# ------------------ FIXED DISORDER SELECTOR (NO 'nan' OPTION) ------------------
disorder_options = [d for d in le_disorder.classes_ if str(d).lower() != "nan"]

disorder_input = st.selectbox(
    "Any disorder symptoms?",
    disorder_options
)

# Safe transform
if disorder_input not in le_disorder.classes_:
    disorder_input = "None"

# ------------------------------ SLEEP DURATION ------------------------------
bt = datetime.datetime.combine(datetime.date.today(), bedtime)
wt = datetime.datetime.combine(datetime.date.today(), wakeuptime)
if wt < bt:
    wt += datetime.timedelta(days=1)

sleep_duration = round((wt - bt).total_seconds()/3600, 2)
st.info(f"🕒 Estimated Sleep Duration: **{sleep_duration} hours**")

# Age range check
enough, ideal_hours = sleep_enough_by_age(age, sleep_duration)
min_hours = int(ideal_hours.split("-")[0])
max_hours = int(ideal_hours.split("-")[1].split()[0])

# ------------------------------ ENCODE INPUTS ------------------------------
med_val = le_meditation.transform([meditation])[0]
con_val = le_consistency.transform([consistency])[0]
dis_val = le_disorder.transform([disorder_input])[0]

input_features = np.array([[age, med_val, con_val, sleep_duration, stress, dis_val]])

# ------------------------------ MAIN PREDICTION BUTTON ------------------------------
if st.button("✨ Analyze My Sleep"):
    
    # AI prediction — Sleep Quality
    pred_quality = model_quality.predict(input_features)
    quality = le_quality.inverse_transform(pred_quality)[0]

    # Adjust quality
    if sleep_duration < min_hours:
        quality = "Poor" if sleep_duration < min_hours - 1 else "Average"
    elif sleep_duration > max_hours:
        quality = "Average"

    # Sleep Score
    sleep_score = 50

    if quality == "Excellent":
        sleep_score += 40
    elif quality == "Average":
        sleep_score += 20
    
    if sleep_duration < min_hours:
        sleep_score -= (min_hours - sleep_duration) * 5
    elif sleep_duration > max_hours:
        sleep_score -= (sleep_duration - max_hours) * 5

    sleep_score -= stress * 2
    if meditation == "Yes":
        sleep_score += 5
    if consistency == "Yes":
        sleep_score += 5
    if disorder_input != "None":
        sleep_score -= 10

    sleep_score = max(0, min(100, int(sleep_score)))

    st.markdown("---")
    st.success(f"🌙 Predicted Sleep Quality: **{quality.upper()}**")
    st.info(f"🛏 AI Sleep Score: **{sleep_score}/100**")

    # Predict Recommendation Category
    pred_rec = model_rec.predict(input_features)
    rec_category = le_rec.inverse_transform(pred_rec)[0]

    rec_map = {
        'Caffeine': ['Reduce caffeine intake', 'Avoid caffeine at night'],
        'Meditation': ['Meditate 10–20 mins daily', 'Practice deep breathing'],
        'Exercise': ['Light exercise daily', 'Yoga for sleep'],
        'Routine': ['Fix your sleep schedule', 'Avoid late-night screens'],
        'Stress': ['Stress-relief breathing exercises', 'Avoid heavy work before bed']
    }

    st.markdown("### 💡 AI Recommendations:")
    for r in rec_map[rec_category]:
        st.markdown(f"🌝 {r}")

    # ---------------------- QUOTES SECTION ----------------------
quotes = [
    "Your future depends on your dreams—so go to sleep.",
    "Sleep is the best meditation.",
    "A well-rested mind is a powerful mind.",
    "Good sleep is the foundation of a healthy life.",
    "Your body heals when you sleep.",
    "Let today’s worries drift away with tonight’s dream.",
    "Sleep because your body loves you.",
    "Every good day starts the night before.",
    "Rest is not a waste of time; it’s an investment.",
    "The best bridge between despair and hope is a good night’s sleep."
]

random_quote = np.random.choice(quotes)

st.markdown("## 🌟 Sleep Inspiration Quote")
st.markdown(f"💫 *{random_quote}*")
