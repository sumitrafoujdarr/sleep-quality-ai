import streamlit as st
from auth import authentication_controller, logout_button
from sleep_app import show_sleep_app

st.set_page_config(page_title="Sleep Analyzer", page_icon="🌙")

# AUTHENTICATION SYSTEM
logged_in = authentication_controller()

# IF LOGGED IN → SHOW SLEEP ANALYZER
if logged_in:
    logout_button()
    show_sleep_app()
