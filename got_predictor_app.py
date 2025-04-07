# datei: got_predictor_app.py

import streamlit as st
import pandas as pd
import joblib  # falls du dein Modell speicherst
import pickle

# Modell und Spalten laden
model = pickle.load(open("model.pkl", "rb"))
feature_cols = pickle.load(open("feature_columns.pkl", "rb"))

# Beispielhafte Inputs
st.title("🧙 Game of Thrones – Charakter überlebens-Check")

st.markdown("Erstelle deinen Charakter und finde heraus, ob er überlebt!")

male = st.selectbox("Geschlecht", ["Männlich", "Weiblich"]) == "Männlich"
age = st.slider("Alter", 0, 100, 30)
is_noble = st.checkbox("Adeliger?")
is_married = st.checkbox("Verheiratet?")
num_dead_relatives = st.slider("Tote Angehörige", 0, 20, 0)
house = st.selectbox("Haus", ["House Stark", "House Lannister", "House Targaryen", "Other"])
culture = st.selectbox("Kultur", ["Northmen", "Ironborn", "Andal", "Other"])
allegiance = st.selectbox("Allegiance", ["Stark", "Lannister", "Other"])

# Eingabe-Feature-Vektor bauen
input_dict = {
    "male": int(male),
    "isNoble": int(is_noble),
    "isMarried": int(is_married),
    "numDeadRelations": num_dead_relatives,
    "age_filled": age,
    "has_age": 1,  # weil wir das Alter angegeben haben
    "noble_and_married": int(is_noble and is_married),
    "house_grouped_" + (house if house in ["House Stark", "House Lannister", "House Targaryen"] else "Other"): 1,
    "culture_" + (culture if culture in ["Northmen", "Ironborn", "Andal"] else "Other"): 1,
    "allegiance_grouped_" + (allegiance if allegiance in ["Stark", "Lannister"] else "Other"): 1,
}

# Alle Features auf 0 initialisieren
input_df = pd.DataFrame([0] * len(feature_cols), index=feature_cols).T

# Dann nur setzen, was im input_dict drin ist
for k, v in input_dict.items():
    if k in input_df.columns:
        input_df[k] = v

if st.button("Vorhersage starten"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0]

    st.subheader("🔮 Ergebnis")
    st.write(f"**Wahrscheinlichkeit zu überleben:** {prob[1]*100:.1f}%")
    if prediction == 1:
        st.success("🎉 Dein Charakter wird wahrscheinlich überleben!")
    else:
        st.error("💀 Dein Charakter hat schlechte Karten...")

