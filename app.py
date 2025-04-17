import os
import pickle
import pandas as pd
import streamlit as st

# === Seitenstruktur ===
image_pages = {
    "Feature Importance": ("feature_importance.png", "Welche Merkmale haben den größten Einfluss auf die Vorhersage?"),
    "Confusion Matrix": ("heatmap.png", "Tatsächlich vs. Vorhergesagt: Wie gut unterscheidet das Modell zwischen Tot und Lebendig?"),
    "ROC-Kurve": ("roc.png", "Bewertet die Modellgüte über verschiedene Schwellen. AUC = Fläche unter der Kurve."),
    "Überlebenswahrscheinlichkeit Histogramm": ("survival_probability_hist.png", "Wie sicher ist sich das Modell bei der Prognose?"),
    "Überleben nach Geschlecht": ("survival_by_gender.png", "Vergleich zwischen männlichen und weiblichen Charakteren."),
    "Überleben nach Adel": ("survival_by_nobility.png", "Haben Adelige bessere Chancen?"),
    "Überleben nach Alter (Histogramm)": ("survival_by_age.png", "Welche Altersgruppen überleben am häufigsten?"),
    "Überleben toter Verwandter": ("survival_has_dead_relatives.png", "Einfluss toter Verwandter auf Überlebenschance."),
    "Überleben Heirat": ("survival_by_isMarried.png", "Verheiratete vs. Unverheiratete."),
    "Klassenverteilung": ("klassenverteilung.png", "Wie viele leben? Wie viele sind tot?"),
    "Partial Dependence Plot (PDP)": ("partial_dependence.png", "Einfluss einzelner Features auf die Vorhersage."),
    "Kumulative Feature-Wichtigkeit": ("kumulative-feature-wichtigkeit.png", "Wie viel erklärt man mit wenigen Features?"),
    "Korrelation zwischen Features": ("feature-korrelationen.png", "Zusammenhänge zwischen Features erkennen."),
    "Beispielbaum aus dem Random Forest": ("random_forest1.png", "Wie trifft das Modell Entscheidungen?"),
    "Charaktertypen (Cluster)": ("cluster.png", "Cluster von Figuren nach ähnlichen Eigenschaften.")
}

slug_to_title = {
    "vorhersage": "Überlebenschance",
    "feature-importance": "Feature Importance",
    "confusion-matrix": "Confusion Matrix",
    "roc-kurve": "ROC-Kurve",
    "histogramm": "Überlebenswahrscheinlichkeit Histogramm",
    "geschlecht": "Überleben nach Geschlecht",
    "adel": "Überleben nach Adel",
    "alter": "Überleben nach Alter (Histogramm)",
    "tote-verwandte": "Überleben toter Verwandter",
    "heirat": "Überleben Heirat",
    "klassenverteilung": "Klassenverteilung",
    "pdp": "Partial Dependence Plot (PDP)",
    "kumulative-wichtigkeit": "Kumulative Feature-Wichtigkeit",
    "korrelation": "Korrelation zwischen Features",
    "baum": "Beispielbaum aus dem Random Forest",
    "cluster": "Charaktertypen (Cluster)"
}
title_to_slug = {v: k for k, v in slug_to_title.items()}

# === Konfiguration ===
st.set_page_config(page_title="GoT Dashboard", layout="wide")
st.title("Game of Thrones Modell-Dashboard")

# === Hilfsfunktionen ===
def show_saved_image(filename, width=None):
    filepath = os.path.join("public/pictures", filename)
    if os.path.exists(filepath):
        st.image(filepath, width=width)
    else:
        st.warning(f"Bild nicht gefunden: {filename}")

def load_model_and_features():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f:
        features = pickle.load(f)
    return model, features

# === Modell & Features global laden ===
model, feature_names = load_model_and_features()


# === Seitenlogik ===
query_params = st.query_params
current_page = query_params.get("page", None)

if current_page is None:
    st.subheader("Übersicht aller Modell-Visualisierungen")
    cols = st.columns(3)
    for i, (title, (img, desc)) in enumerate(image_pages.items()):
        slug = title_to_slug[title]
        with cols[i % 3]:
            st.image(os.path.join("public/pictures", img), caption=title, use_container_width=True)
            st.markdown(f"[Details ansehen](?page={slug})")

elif current_page == "vorhersage":
    st.subheader("Überlebenschance deines Charakters")
    st.markdown("**Erstelle deinen Charakter und erhalte eine Vorhersage.**")

    model, feature_names = load_model_and_features()

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

    input_dict = {
        "male": int(male),
        "age": age,
        "isNoble": int(is_noble),
        "isMarried": int(is_married),
        "numDeadRelations": num_dead_relatives,
    }

    def safe_onehot(prefix, value):
        col = f"{prefix}_{value}"
        return col if col in feature_names else f"{prefix}_Other"

    for cat, val in [("house", house), ("culture", culture), ("allegiances", allegiance), ("title", title)]:
        input_dict[safe_onehot(cat, val)] = 1

    input_df = pd.DataFrame([0] * len(feature_names), index=feature_names).T
    for k, v in input_dict.items():
        if k in input_df.columns:
            input_df[k] = v

    if st.button("Überlebenschance berechnen"):
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0]
        st.subheader("Prognose")
        st.markdown(f"**Wahrscheinlichkeit zu überleben:** {prob[1]*100:.1f}%")
        if prediction == 1:
            st.success("Dein Charakter hat gute Überlebenschancen.")
        else:
            st.error("💀 Dein Charakter wird vermutlich nicht überleben.")

elif current_page in slug_to_title:
    title = slug_to_title[current_page]
    st.markdown(f"## {title}")
    st.caption(image_pages[title][1])
    
    # Bild anzeigen
    if image_pages[title][0]:
        show_saved_image(image_pages[title][0])

    # Erweiterte Inhalte je nach Visualisierung
    if title == "Feature Importance":
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False)
        st.markdown("**Top-Merkmale nach Bedeutung für das Modell:**")
        st.dataframe(importance_df.reset_index(drop=True), use_container_width=True)

    elif title == "Confusion Matrix":
        st.markdown("""
        Die Confusion Matrix zeigt die Treffer des Modells:
        - **Oben links**: korrekt als "tot" erkannt (True Negative)
        - **Unten rechts**: korrekt als "lebendig" erkannt (True Positive)
        - **Oben rechts**: fälschlich als "lebendig" klassifiziert (False Positive)
        - **Unten links**: fälschlich als "tot" klassifiziert (False Negative)
        """)

    elif title == "ROC-Kurve":
        st.markdown("""
        Die ROC-Kurve stellt die Modellgüte dar. Die **AUC (Area Under Curve)** beschreibt die Trennschärfe:
        - AUC = 0.5 → zufällig
        - AUC = 1.0 → perfekt
        Je näher an 1, desto besser das Modell.
        """)

    elif title == "Überlebenswahrscheinlichkeit Histogramm":
        st.markdown("""
        Dieses Histogramm zeigt, wie sicher sich das Modell bei seinen Vorhersagen ist.
        - Ein Peak bei **0.5** bedeutet Unsicherheit.
        - Verteilung bei **0** und **1** zeigt, dass das Modell klare Aussagen trifft.
        """)

    elif title == "Partial Dependence Plot (PDP)":
        st.markdown("""
        Der PDP zeigt, wie sich ein einzelnes Merkmal – z. B. das Alter – auf die Vorhersage auswirkt, **unabhängig vom Rest** der Daten.
        So kannst du sehen, ob z. B. ein höheres Alter eher mit „tot“ oder „überlebt“ zusammenhängt.
        """)

    elif title == "Kumulative Feature-Wichtigkeit":
        st.markdown("""
        Diese Grafik zeigt, wie viel der Modellleistung durch die wichtigsten Features erklärt wird.
        Man erkennt z. B., ob 5 oder 20 Features den Großteil der Erklärung liefern.
        """)

    elif title == "Korrelation zwischen Features":
        st.markdown("""
        Korrelation bedeutet: **Wie stark hängen zwei Merkmale zusammen?**
        - **Rot = starke positive Korrelation** (beide steigen gemeinsam)
        - **Blau = negative Korrelation** (eines steigt, das andere sinkt)
        Hohe Korrelation kann auf redundante Features hinweisen.
        """)

    elif title == "Beispielbaum aus dem Random Forest":
        st.markdown("""
        Ein einzelner Entscheidungsbaum aus dem Random Forest.
        Er zeigt:
        - Welche Merkmale zuerst geprüft werden
        - Wie viele Daten an jedem Punkt „links“ oder „rechts“ laufen
        - Wie die Entscheidung „tot“ oder „lebt“ getroffen wird
        """)

    elif title == "Charaktertypen (Cluster)":
        st.markdown("""
        Hier wurden Figuren automatisch zu Gruppen geclustert (ähnliche Eigenschaften).
        Farben zeigen unterschiedliche Cluster.
        Die Positionen basieren auf einer Reduktion der Merkmale in 2D (PCA oder t-SNE).
        """)

    elif title == "Klassenverteilung":
        st.markdown("""
        Diese Grafik zeigt, wie **unausgeglichen** die Daten sind:
        - Wenn z. B. viel mehr überleben als sterben (oder umgekehrt), kann das Modell verzerrt sein.
        - Deshalb ist **SMOTE** als Ausgleich wichtig.
        """)

    elif title == "Überleben nach Geschlecht":
        st.markdown("""
        Vergleich der Überlebensraten zwischen männlichen und weiblichen Charakteren.
        Gibt Hinweise auf implizite Vorurteile im Modell oder in den Daten.
        """)

    elif title == "Überleben nach Adel":
        st.markdown("""
        Überleben Adelige öfter? Diese Analyse zeigt den Zusammenhang zwischen Adelstitel und Überlebensrate.
        """)

    elif title == "Überleben Heirat":
        st.markdown("""
        Gibt es einen Zusammenhang zwischen Familienstand und Überleben?
        Diese Grafik zeigt, ob verheiratete Charaktere anders abschneiden.
        """)

    elif title == "Überleben nach Alter (Histogramm)":
        st.markdown("""
        Wie verteilt sich das Alter bei Überlebenden und Toten?
        Diese Verteilung hilft zu erkennen, ob z. B. junge Figuren bevorzugt überleben.
        """)

    elif title == "Überleben toter Verwandter":
        st.markdown("""
        Charaktere mit toten Verwandten könnten ein höheres Risiko tragen.
        Diese Grafik untersucht genau das.
        """)

# === Sidebar Navigation ===
st.sidebar.markdown("## Schnellauswahl")

# Aktuelle Seite aus den Query-Params ermitteln
current_slug = st.query_params.get("page", "start")
if isinstance(current_slug, list):
    current_slug = current_slug[0]

# Den zur aktuellen Seite passenden Titel finden
selected_title = slug_to_title.get(current_slug, "Start")

# Seitenliste in der gewünschten Reihenfolge
seitenliste = ["Start", "Überlebenschance"] + list(image_pages.keys())

# Selectbox anzeigen mit dem aktuell ausgewählten Titel
sidebar_choice = st.sidebar.selectbox("Seite", seitenliste, index=seitenliste.index(selected_title))

# Nur rerun auslösen, wenn sich die Auswahl aktiv geändert hat
if sidebar_choice == "Start" and current_slug != "start":
    st.query_params.clear()
    st.rerun()
elif sidebar_choice == "Überlebenschance" and current_slug != "vorhersage":
    st.query_params.update({"page": "vorhersage"})
    st.rerun()
elif sidebar_choice != selected_title:
    st.query_params.update({"page": title_to_slug[sidebar_choice]})
    st.rerun()
