# ----------------------------
# PREDICTION
# ----------------------------
if st.button("✨ Analyze My Sleep"):
    input_data = np.array([[age, sleep_duration, med_val, con_val, dis_val, sleep_enough]])
    pred_quality = model.predict(input_data)
    sleep_quality = le_quality.inverse_transform(pred_quality)[0]  # Will be Excellent / Average / Poor

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
    if sleep_quality.lower() == "poor":
        st.error("😴 Your sleep quality seems poor. Follow recommendations!")
        st.markdown("""
        <div class='recommendation'>
        🔹 Avoid screens 30 mins before sleep<br>
        🔹 No caffeine after evening<br>
        🔹 Meditate daily 15–25 mins<br>
        🔹 Light exercise regularly<br>
        🔹 Avoid stress before bed<br>
        🔹 Ensure you get enough sleep hours for your age
        </div>
        """, unsafe_allow_html=True)
    elif sleep_quality.lower() == "average":
        st.info("💤 Your sleep is average — moderate improvements recommended!")
        st.markdown("""
        <div class='recommendation'>
        🔹 Maintain consistent bedtime<br>
        🔹 Reduce late-night screen exposure<br>
        🔹 Drink water before bed, avoid heavy meals<br>
        🔹 Meditate 10–15 mins<br>
        🔹 Check if your sleep duration meets age recommendations
        </div>
        """, unsafe_allow_html=True)
    else:  # Excellent
        st.success("🌟 Excellent Sleep Quality — Keep it up!")
        st.markdown("""
        <div class='recommendation'>
        🔹 Maintain healthy routine<br>
        🔹 Avoid overworking late nights<br>
        🔹 Stay hydrated and stress-free<br>
        🔹 Continue mindfulness & balance<br>
        🔹 Your sleep duration is sufficient — keep it consistent!
        </div>
        """, unsafe_allow_html=True)
