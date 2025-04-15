import streamlit as st
import pandas as pd
import pickle

# Streamlit-Config anpassen
st.set_page_config(
    page_title="Game of Thrones – Überlebensprognose",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom Styling (dunkler Stil + GoT-Flair)
st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Georgia', serif;
            background-color: #1e1e1e;
            color: #f0f0f0;
        }
        .stButton button {
            background-color: #5c3d2e;
            color: #fff;
            border-radius: 0.3rem;
            padding: 0.6em 1.2em;
            font-weight: bold;
            border: none;
        }
        .stSlider > div {
            color: #f0f0f0;
        }
        .stSelectbox label, .stCheckbox label {
            color: #cccccc;
        }
        h1, h2, h3, .stSubheader {
            color: #d6b676;
        }
    </style>
""", unsafe_allow_html=True)

# Titel
st.title("Game of Thrones – Überlebensprognose")
st.markdown("**Erstelle deinen Charakter und erfahre, ob er in Westeros überlebt.**")

# Modell laden
model = pickle.load(open("model.pkl", "rb"))
feature_cols = pickle.load(open("feature_columns.pkl", "rb"))

# Eingabe-Elemente
st.header("Charakter erstellen")

col1, col2 = st.columns(2)

with col1:
    male = st.selectbox("Geschlecht", ["Männlich", "Weiblich"]) == "Männlich"
    age = st.slider("Alter", 0, 100, 30)
    is_noble = st.checkbox("Adelig?")
    is_married = st.checkbox("Verheiratet?")

with col2:
    num_dead_relatives = st.slider("Tote Angehörige", 0, 20, 0)
    house = st.selectbox("Haus", ["House Stark", "House Lannister", "House Targaryen", "Other"])
    culture = st.selectbox("Kultur", ["Northmen", "Ironborn", "Andals", "Other"])
    allegiance = st.selectbox("Treue zu", ["House Stark", "House Lannister", "House Targaryen", "Other"])
    title = st.selectbox("Titel", ["Ser", "Lord", "Lady", "Maester", "King", "Queen", "Other"])

# Eingabedaten vorbereiten
input_dict = {
    "male": int(male),
    "age": age,
    "isNoble": int(is_noble),
    "isMarried": int(is_married),
    "numDeadRelations": num_dead_relatives,
}

def safe_onehot(prefix, value):
    col = f"{prefix}_{value}"
    return col if col in feature_cols else f"{prefix}_Other"

input_dict[safe_onehot("house", house)] = 1
input_dict[safe_onehot("culture", culture)] = 1
input_dict[safe_onehot("allegiances", allegiance)] = 1
input_dict[safe_onehot("title", title)] = 1

# DataFrame vorbereiten
input_df = pd.DataFrame([0]*len(feature_cols), index=feature_cols).T
for k, v in input_dict.items():
    if k in input_df.columns:
        input_df[k] = v

# Vorhersage
if st.button("Überlebenschance berechnen"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0]

    st.subheader("Prognose")
    st.markdown(f"**Wahrscheinlichkeit zu überleben:** {prob[1]*100:.1f}%")

    if prediction == 1:
        st.success("Dein Charakter hat gute Überlebenschancen.")
    else:
        st.error("Dein Charakter wird vermutlich nicht überleben.")

