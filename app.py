import streamlit as st
import pickle
import datetime

st.title("🧘‍♀️ AI Sleep Quality Predictor 😴")
st.write("Enter your details to check your sleep quality and get recommendations.")

# Inputs
age = st.number_input("Your Age", min_value=10, max_value=100)
bedtime = st.time_input("Bedtime (24-hour format)")
wakeuptime = st.time_input("Wakeup Time (24-hour format)")
meditation = st.checkbox("Do you meditate regularly?")
consistency = st.checkbox("Do you maintain a consistent sleep schedule?")

# Sleep Duration Calculation
bed = datetime.datetime.combine(datetime.date.today(), bedtime)
wake = datetime.datetime.combine(datetime.date.today(), wakeuptime)
if wake < bed:
    wake += datetime.timedelta(days=1)
sleep_duration = round((wake - bed).seconds / 3600, 2)

st.write(f"🕒 Calculated Sleep Duration: {sleep_duration} hours")

if st.button("Predict Sleep Quality"):
    model = pickle.load(open("sleep_model.pkl", "rb"))
    med = 1 if meditation else 0
    cons = 1 if consistency else 0
    pred = model.predict([[age, sleep_duration, med, cons]])[0]
    labels = ["Average", "Excellent", "Poor"]
    quality = labels[pred]

    # Sleep score (for display)
    score = int((sleep_duration / 8) * 100)
    if meditation: score += 5
    if consistency: score += 5
    if score > 100: score = 100

    st.subheader(f"🌙 Your Sleep Quality: {quality}")
    st.write(f"💤 Sleep Score: {score}/100")

    # Recommendations
    st.markdown("### 🌼 Recommendations:")
    if quality == "Poor":
        st.write("- Try sleeping at the same time daily")
        st.write("- Avoid screens 1 hour before bed")
        st.write("- Try short meditation before sleep")
    elif quality == "Average":
        st.write("- Keep consistency and meditation routine")
        st.write("- Avoid caffeine late evening")
    else:
        st.write("- Great job! Maintain your sleep routine and meditation habit 😊")
