import streamlit as st
import pandas as pd

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="LIAR - Trustworthiness Predictor",
    page_icon="⚖️",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&display=swap');

.stApp {
    background: radial-gradient(
        circle at center,
        #91dcff 0%,
        #f5bdff 90%,
        #e173f5 195%
    );
}

.block-container {
    padding-top: 5rem;
    padding-bottom: 4rem;
    max-width: 1200px;
}

.hero-title {
    text-align: center;
    font-size: 4rem;
    font-weight: 800;
    color: #312e81;
}

.hero-subtitle {
    text-align: center;
    font-family: 'Orbitron', sans-serif;
    font-size: 1.3rem;
    color: #4c1d95;
    margin-bottom: 2rem;
}

.section-title {
    font-family: 'Orbitron', sans-serif;
    text-align: center;
    font-size: 2rem;
    font-weight: 700;
    color: #312e81;
    margin-top: 2rem;
    margin-bottom: 1rem;
}

.metric-card {
    background-color: rgba(255,255,255,0.8);
    padding: 1rem;
    border-radius: 12px;
    border: 1px solid #c4b5fd;
}

/* Metric card */
[data-testid="stMetric"] {
    width: 100%;
    background-color: rgba(255,255,255,0.0.85);
    border: 2px solid #7c3aed;
    padding: 15px;
    border-radius: 15px;
    box-shadow:0 4px 12px rgba(0,0,0,0.08);
}

/* Metric value */
[data-testid="stMetricValue"] {
    color: #4c1d95;
    font-size: 2rem;
}

/* Metric label */
[data-testid="stMetricLabel"] {
    color: #312e81;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# HERO SECTION
# =====================================================

# Optional banner image
# st.image("assets/banner.png", use_container_width=True)

col1, col2, col3 = st.columns([1,3,1])

with col1:
    st.markdown("""
    <div style='text-align:center'>
        <div style='font-size:1rem;'>Trained on</div>
        <div style='font-size:2rem; font-weight:700;'>35k+</div>
        <div style='font-size:1rem;'>real political statements</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
    <div style='text-align:center'>
        <div style='font-size:1rem;'>Made by</div>
        <div style='font-size:2rem; font-weight:700;'>1000+</div>
        <div style='font-size:1rem;'>different speakers</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style='text-align:center'>
        <div style='font-size:1rem;'>Preditions of</div>
        <div style='font-size:2rem; font-weight:700;'>3</div>
        <div style='font-size:1rem;'>different trust levels</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
    <div style='text-align:center'>
        <div style='font-size:1rem;'>Made by</div>
        <div style='font-size:2rem; font-weight:700;'>3</div>
        <div style='font-size:1rem;'>different models</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.image(
        "/home/mnavarrosotes/code/navaso93/final_project/liar_project/images/logo_liar.png",
        use_container_width=True,
    )

    st.markdown(
        '<div class="hero-subtitle">Detect Trustworthy, Questionable and Unreliable Political Claims</div>',
        unsafe_allow_html=True
    )


st.divider()

# =====================================================
# DATASET INSIGHTS
# =====================================================

st.markdown(
    '<div class="section-title">Dataset Insights</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    st.subheader("Top Unreliable Speakers")

    st.dataframe(
        pd.DataFrame({
            "Speaker": ["Speaker A", "Speaker B", "Speaker C"],
            "Statements": [120, 105, 99],
            "% Unreliable": [72, 69, 65]
        }),
        use_container_width=True
    )

with col2:

    st.subheader("Top Trustworthy Speakers")

    # Here we input a dataframe from the backend with top5 trustworthy speakers (True Statements / Total Statements)
    st.dataframe(
        pd.DataFrame({
            "Speaker": ["Speaker X", "Speaker Y", "Speaker Z"],
            "Statements": [85, 79, 70],
            "% Trustworthy": [82, 80, 78]
        }),
        use_container_width=True
    )

st.subheader("Context Statistics")

st.dataframe(
    pd.DataFrame({
        "Context": [
            "Campaign Rally",
            "TV Interview",
            "Debate"
        ],
        "Statements": [
            400,
            320,
            290
        ]
    }),
    use_container_width=True
)

# =====================================================
# INPUT SECTION
# =====================================================

st.markdown(
    '<div class="section-title">Analyze a Statement</div>',
    unsafe_allow_html=True
)

statement = st.text_area(
    "Statement",
    height=150,
    placeholder="Enter a statement made by a political figure..."
)

col1, col2 = st.columns(2)

with col1:
    speaker = st.text_input(
        "Speaker",
        placeholder="Enter the speaker full first and last name"
    )

with col2:
    context = st.text_input(
        "Context",
        placeholder="Enter the context (press, speech, social media...)"
    )

analyze_button = st.button(
    "Predict statement trustworthiness",
    use_container_width=True
)

# =====================================================
# PREDICTION RESULTS
# =====================================================

st.markdown(
    '<div class="section-title">Prediction Results</div>',
    unsafe_allow_html=True
)

# only show after prediction
if analyze_button:

    prediction = "Unreliable"

    st.success(
        f"Predicted Label: {prediction}"
    )

    model_results = pd.DataFrame({
        "Model": [
            "Naive Bayes",
            "NB + XGBoost",
            "RoBERTa"
        ],
        "Prediction": [
            "Questionable",
            "Unreliable",
            "Unreliable"
        ],
        "Confidence": [
            0.62,
            0.81,
            0.78
        ]
    })

    st.dataframe(
        model_results,
        use_container_width=True
    )

# =====================================================
# RAG RESULTS
# =====================================================

st.markdown(
    '<div class="section-title">Similar Verified Statements (RAG)</div>',
    unsafe_allow_html=True
)

if analyze_button:

    for i in range(3):

        with st.container(border=True):

            st.markdown(
                f"### Similar Statement {i+1}"
            )

            st.write(
                "Example retrieved statement..."
            )

            st.write(
                "**Speaker:** Example Speaker"
            )

            st.write(
                "**Context:** Example Context"
            )

            st.write(
                "**Label:** Unreliable"
            )

# =====================================================
# LLM ANALYSIS
# =====================================================

st.markdown(
    '<div class="section-title">AI Analysis</div>',
    unsafe_allow_html=True
)

if analyze_button:

    st.markdown(
        """
        Gemini explanation appears here.

        It will explain:

        - Why the statement was classified this way
        - Similar examples found
        - Confidence interpretation
        - Possible rhetorical patterns
        """
    )

# =====================================================
# OPTIONAL IMAGE PLACEHOLDERS
# =====================================================

# st.image("assets/model_pipeline.png")
# st.image("assets/rag_diagram.png")
# st.image("assets/confusion_matrix.png")
