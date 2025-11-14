import streamlit as st
import requests

# ---------------- Base Config ----------------
API_URL = 'https://sentiment-analysis-app-1-c2ar.onrender.com'
st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬", layout="centered")

# --------------- Custom CSS for styling ----------------
def set_custom_style(theme="neutral"):
    if theme == "positive":
        st.markdown(
            """
            <style>
            body {background-color: #f6fff9;}
            .main {background-color: #e8fff0; border-radius: 15px; padding: 25px;}
            h1 {color: #1b8e4b; text-align: center;}
            .stTextArea textarea {border: 2px solid #1b8e4b; border-radius: 10px;}
            .stButton>button {background-color: #1b8e4b; color: white; border-radius: 10px;}
            .stButton>button:hover {background-color: #157a3c;}
            </style>
            """,
            unsafe_allow_html=True,
        )
    elif theme == "negative":
        st.markdown(
            """
            <style>
            body {background-color: #1b1b1b;}
            .main {background-color: #2a2a2a; border-radius: 15px; padding: 25px;}
            h1 {color: #ff4b4b; text-align: center;}
            .stTextArea textarea {border: 2px solid #ff4b4b; border-radius: 10px; background-color: #333; color: white;}
            .stButton>button {background-color: #ff4b4b; color: white; border-radius: 10px;}
            .stButton>button:hover {background-color: #d33a3a;}
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:  # neutral/default
        st.markdown(
            """
            <style>
            body {background-color: #f8f9fa;}
            .main {background-color: #ffffff; border-radius: 15px; padding: 25px;}
            h1 {color: #444; text-align: center;}
            .stTextArea textarea {border: 2px solid #ccc; border-radius: 10px;}
            .stButton>button {background-color: #007bff; color: white; border-radius: 10px;}
            .stButton>button:hover {background-color: #0069d9;}
            </style>
            """,
             unsafe_allow_html=True,
        )

# Initial theme before result
set_custom_style("neutral")

# ---------------- Streamlit UI ----------------
st.title("💬 Sentiment Analysis of Text")

user_input = st.text_area(
    label="Enter your comment below:",
    placeholder="Type your comment here...",
    height=100
)

analyze_button = st.button("🔍 Analyze Sentiment")

# ---------------- Main Control ----------------
if analyze_button:
    if user_input.strip():
        st.write('✨ Processing your text through the API...')
        payload = {'review': user_input}
        try:
            with st.spinner('Analyzing...'):
                response = requests.post(API_URL, json=payload)

            if response.status_code == 200:
                response_data = response.json()
                sentiment = response_data.get('sentiment', '').capitalize()

                if sentiment == 'Positive':
                    set_custom_style("positive")
                    st.balloons()
                    st.markdown("<h2 style='color:#1b8e4b;text-align:center;'>Prediction: Positive 👍</h2>", unsafe_allow_html=True)
                elif sentiment == 'Negative':
                    set_custom_style("negative")
                    st.markdown("<h2 style='color:#ff4b4b;text-align:center;'>Prediction: Negative 👎</h2>", unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Could not determine the sentiment. Try a different review.")
            else:
                try:
                    error_details = response.json()
                    st.error(f"API Error: {error_details.get('error', 'An unknown error occurred.')}")
                except requests.exceptions.JSONDecodeError:
                    st.error(f"API Error: Status code {response.status_code} - {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("🚫 Connection Error: Could not connect to Flask API. Please make sure it's running.")
        except Exception as e:
            st.error(f"⚠️ Unexpected error: {e}")
    else:
        st.warning('✏️ Please enter a comment to analyze.')
else:
    st.info('👆 Enter a comment and click **Analyze Sentiment** to begin.')

