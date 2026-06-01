import requests
import streamlit as st

st.set_page_config(page_title="Quote Classifier", page_icon="💬")

st.title("Zitat-Klassifizierung")

# --- Eingabefelder ---
quote = st.text_area("Zitat", placeholder="Gib hier dein Zitat ein...")

col1, col2, col3 = st.columns(3)
with col1:
    add1 = st.text_input("Additional information 1", "")
with col2:
    add2 = st.text_input("Additional information 2", "")
with col3:
    add3 = st.text_input("Additional information 3", "")

# Statusanzeige-Platzhalter
status_placeholder = st.empty()
result_placeholder = st.empty()

# Mapping Kategorie -> Farbe
CATEGORY_COLORS = {
    1: "#ff4b4b",  # rot
    2: "#ff914d",  # orange
    3: "#ffd93b",  # gelb
    4: "#9ae66e",  # hellgrün
    5: "#2ecc71",  # grün
}

def render_category(category: int):
    color = CATEGORY_COLORS.get(category, "#cccccc")
    st.markdown(
        f"""
        <div style="
            border-radius: 8px;
            padding: 16px;
            background-color: {color};
            color: #000000;
            font-weight: 600;
            text-align: center;
            font-size: 20px;
        ">
            Kategorie: {category}
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- API Call ---
if st.button("Analysieren"):
    if not quote.strip():
        status_placeholder.error("Bitte zuerst ein Zitat eingeben.")
    else:
        status_placeholder.info("Sende Anfrage an API...")
        try:
            # Beispiel: API-URL anpassen
            api_url = "https://example.com/api/classify"

            payload = {
                "quote": quote,
                "additional_information_1": add1 or None,
                "additional_information_2": add2 or None,
                "additional_information_3": add3 or None,
            }

            response = requests.post(api_url, json=payload, timeout=15)
            if response.status_code == 200:
                data = response.json()
                # Erwartet: data["category"] ist eine Zahl 1–5
                category = int(data.get("category", 0))

                status_placeholder.success("Anfrage erfolgreich.")
                if 1 <= category <= 5:
                    result_placeholder.write("Ergebnis der Klassifizierung:")
                    render_category(category)
                else:
                    result_placeholder.error(
                        f"Ungültige Kategorie aus API erhalten: {category}"
                    )
            else:
                status_placeholder.error(
                    f"API-Fehler: HTTP {response.status_code}"
                )
        except Exception as e:
            status_placeholder.error(f"Fehler bei der API-Anfrage: {e}")
