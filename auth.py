import streamlit as st

# Initialize session state only once
def init_session():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if "users" not in st.session_state:
        # Default demo users
        st.session_state.users = {
            "sumitra": "1234",
            "admin": "admin"
        }

    if "show_signup" not in st.session_state:
        st.session_state.show_signup = False


# -------- SIGNUP PAGE ----------
def signup_page():
    st.title("📝 Create Account")

    new_user = st.text_input("Choose Username")
    new_pass = st.text_input("Choose Password", type="password")
    confirm_pass = st.text_input("Confirm Password", type="password")

    if st.button("Create Account"):
        if new_user == "" or new_pass == "":
            st.error("Please fill all fields.")
        elif new_pass != confirm_pass:
            st.error("Passwords do not match!")
        elif new_user in st.session_state.users:
            st.error("Username already exists!")
        else:
            st.session_state.users[new_user] = new_pass
            st.success("Account created! Please login.")
            st.session_state.show_signup = False

    if st.button("Back to Login"):
        st.session_state.show_signup = False


# -------- LOGIN PAGE ----------
def login_page():
    st.title("🔐 Login to Continue")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in st.session_state.users and st.session_state.users[username] == password:
            st.session_state.logged_in = True
            st.success("Login successful!")
        else:
            st.error("Incorrect username or password")

    if st.button("Create New Account"):
        st.session_state.show_signup = True


# -------- LOGOUT BUTTON ----------
def logout_button():
    if st.button("Logout"):
        st.session_state.logged_in = False


# -------- AUTH CONTROLLER ----------
def authentication_controller():
    init_session()

    if st.session_state.logged_in:
        return True
    else:
        if st.session_state.show_signup:
            signup_page()
        else:
            login_page()
        return False
