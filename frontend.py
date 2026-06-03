"""
What is this file?
This file defines the Streamlit frontend for the LIAR prediction project.

What is its responsibility?
It collects user input, lets the user optionally provide metadata, sends the request to the FastAPI backend, and displays the prediction result.
"""

import requests
import streamlit as st


API_URL = "http://34.22.226.103:8000/predict"

MODEL_OPTIONS = ["naive"]

SUBJECT_OPTIONS = [
    "unknown",
    "health",
    "economy",
    "taxation",
    "budget",
    "education",
    "immigration",
    "elections",
    "security",
    "foreign_affairs",
    "environment",
    "energy",
    "transportation",
    "military",
    "government",
    "social_issues",
    "science",
    "other",
]

JOB_TITLE_OPTIONS = [
    "unknown",
    "president",
    "governor",
    "senator",
    "legislator",
    "mayor",
    "ambassador",
    "representative",
    "legal",
    "media",
    "private executive",
    "candidate",
    "other",
]

STATE_OPTIONS = [
    "unknown",
    "alabama",
    "alaska",
    "arizona",
    "arkansas",
    "california",
    "colorado",
    "connecticut",
    "delaware",
    "florida",
    "georgia",
    "hawaii",
    "idaho",
    "illinois",
    "indiana",
    "iowa",
    "kansas",
    "kentucky",
    "louisiana",
    "maine",
    "maryland",
    "massachusetts",
    "michigan",
    "minnesota",
    "mississippi",
    "missouri",
    "montana",
    "nebraska",
    "nevada",
    "new hampshire",
    "new jersey",
    "new mexico",
    "new york",
    "north carolina",
    "north dakota",
    "ohio",
    "oklahoma",
    "oregon",
    "pennsylvania",
    "rhode island",
    "south carolina",
    "south dakota",
    "tennessee",
    "texas",
    "utah",
    "vermont",
    "virginia",
    "washington",
    "west virginia",
    "wisconsin",
    "wyoming",
    "other",
]

PARTY_OPTIONS = [
    "unknown",
    "republican",
    "democrat",
    "other_party",
    "organization",
    "other",
]

CONTEXT_OPTIONS = [
    "unknown",
    "ad",
    "interview",
    "press release",
    "news conference",
    "debate",
    "social media",
    "statement",
    "email",
    "tv appearance",
    "other",
]


def render_prediction_label(prediction: str) -> None:
    label_styles = {
        "trustworthy": {"background": "#2ecc71", "text": "Trustworthy"},
        "questionable": {"background": "#ffd93b", "text": "Questionable"},
        "unreliable": {"background": "#ff4b4b", "text": "Unreliable"},
    }

    style = label_styles.get(
        prediction,
        {"background": "#cccccc", "text": prediction},
    )

    st.markdown(
        f"""
        <div style="
            border-radius: 8px;
            padding: 18px;
            background-color: {style["background"]};
            color: #000000;
            font-weight: 700;
            text-align: center;
            font-size: 22px;
            margin-top: 12px;
            margin-bottom: 12px;
        ">
            Prediction: {style["text"]}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_probabilities(class_probabilities: dict) -> None:
    st.subheader("Class probabilities")

    for label, probability in class_probabilities.items():
        st.write(f"{label}: {probability:.2%}")
        st.progress(float(probability))


st.set_page_config(
    page_title="LIAR Statement Classifier",
    page_icon="🧠",
    layout="centered",
)

st.title("LIAR Statement Classifier")
st.caption("Naive Bayes MVP with optional metadata and FastAPI backend")

statement = st.text_area(
    "Statement",
    placeholder="Enter a political claim or statement...",
    height=140,
)

model_name = st.selectbox(
    "Choose model",
    MODEL_OPTIONS,
)

use_metadata = st.checkbox("Add optional metadata")

subject = "unknown"
speaker = "unknown"
job_title = "other"
state = "unknown"
party = "unknown"
context = "other"

if use_metadata:
    st.subheader("Optional metadata")

    col1, col2 = st.columns(2)

    with col1:
        subject = st.selectbox("Subject", SUBJECT_OPTIONS)
        job_title = st.selectbox("Job title", JOB_TITLE_OPTIONS)
        party = st.selectbox("Party", PARTY_OPTIONS)

    with col2:
        context = st.selectbox("Context", CONTEXT_OPTIONS)
        state = st.selectbox("State", STATE_OPTIONS)
        speaker = st.text_input("Speaker", value="unknown")


status_placeholder = st.empty()
result_placeholder = st.container()

if st.button("Predict"):
    if not statement.strip():
        status_placeholder.error("Please enter a statement first.")
    else:
        payload = {
            "model_name": model_name,
            "statement": statement,
            "subject": subject,
            "speaker": speaker,
            "job_title": job_title,
            "state": state,
            "party": party,
            "context": context,
        }

        try:
            status_placeholder.info("Sending request to FastAPI backend...")

            response = requests.post(
                API_URL,
                json=payload,
                timeout=30,
            )

            if response.status_code == 200:
                data = response.json()

                status_placeholder.success("Prediction completed.")

                with result_placeholder:
                    render_prediction_label(data["prediction"])

                    st.metric(
                        label="Confidence",
                        value=f"{data['confidence']:.2%}",
                    )

                    render_probabilities(data["class_probabilities"])

                    with st.expander("Raw API response"):
                        st.json(data)

            else:
                status_placeholder.error(
                    f"API error: HTTP {response.status_code}"
                )

                with st.expander("API response"):
                    st.write(response.text)

        except requests.exceptions.ConnectionError:
            status_placeholder.error(
                "Could not connect to the FastAPI backend. Make sure uvicorn is running on http://127.0.0.1:8000."
            )

        except requests.exceptions.Timeout:
            status_placeholder.error("The API request timed out.")

        except Exception as error:
            status_placeholder.error(f"Unexpected error: {error}")
