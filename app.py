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
    "Charaktertypen (Cluster)": ("cluster.png", "Cluster von Figuren nach ähnlichen Eigenschaften."),
    "Überleben nach Haus": ("survival_by_house.png", "Gibt es Häuser mit besonders hoher oder niedriger Überlebensrate?"),
    "Einführungskapitel vs. Überleben": ("intro_chapter_vs_survival.png", "Hängt der Zeitpunkt der Einführung mit dem Überleben zusammen?"),
    "t-SNE Tot vs. Lebendig": ("tsne_survival.png", "Verteilung von Figuren im Merkmalsraum – trennt das Modell Tot und Lebendig?")

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
    "cluster": "Charaktertypen (Cluster)",
    "haus": "Überleben nach Haus",
    "intro": "Einführungskapitel vs. Überleben",
    "tsne": "t-SNE Tot vs. Lebendig"
}
title_to_slug = {v: k for k, v in slug_to_title.items()}

# === Konfiguration ===
st.set_page_config(page_title="GoT Dashboard", layout="wide")
st.title("Game of Thrones Modell-Dashboard")

st.markdown("""
<style>
.card {
    background-color: #1e1e1e;
    border-radius: 15px;
    padding: 1rem;
    height: 350px;            
    display: flex;
    flex-direction: column;
    justify-content: space-between;  
    margin-bottom: 0.5rem;
}

.card img {
    width: 100%;
    height: 180px;
    object-fit: contain;
    border-radius: 4px;
}

.card .card-body {
    flex-grow: 1;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
.card-title {
    font-weight: bold;
    margin-top: 0.5rem;
}
.card-caption {
    font-size: 0.9rem;
    color: #aaa;
    flex-grow: 1;
}
</style>
""", unsafe_allow_html=True)


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

import base64

def get_image_base64(path):
    try:
        with open(path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode("utf-8")
    except:
        return None

# === Modell & Features global laden ===
model, feature_names = load_model_and_features()


# === Seitenlogik ===
query_params = st.query_params
current_page = query_params.get("page", None)

if current_page is None:
    st.subheader("Übersicht aller Modell-Visualisierungen")
    st.markdown("### Visualisierungsgalerie")
    
    cols = st.columns(3)

    for i, (title, (img, desc)) in enumerate(image_pages.items()):
        slug = title_to_slug.get(title, None)
        if slug:
            with cols[i % 3]:
                img_path = os.path.join("public/pictures", img)
                img_data = get_image_base64(img_path)

                if img_data:
                    img_html = f'<img src="data:image/png;base64,{img_data}" alt="{title}"/>'
                else:
                    img_html = '<div style="height:200px; background:#333; color:white; display:flex; align-items:center; justify-content:center;">Bild fehlt</div>'

                st.markdown(f"""
                <div class="card">
                    {img_html}
                    <div class="card-title">{title}</div>
                    <div class="card-caption">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

                if st.button("Details ansehen", key=f"button_{i}"):
                    st.query_params.update({"page": slug})
                    st.rerun()



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
        st.markdown("""
        Interpretation und kritische Einordnung:
                
        - **`book4`**: Figuren, die im 4. Buch erscheinen, überleben häufiger, evtl. weil sie spät eingeführt wurden und dadurch weniger Zeit hatten zu sterben (→ mögliches Data Leakage).
        - **`age`**: Jüngere Charaktere haben offenbar bessere Überlebenschancen. Vielleicht, weil sie weniger in gefährliche Konflikte verwickelt sind.
        - **`male`, `isNoble`, `numDeadRelations`**: Geschlecht, Adel und tote Verwandte beeinflussen die Prognose. Das kann reale Story-Muster widerspiegeln, birgt aber auch das Risiko, Vorurteile zu übernehmen.
        - **`house_unknown`, `culture_unknown`, `allegiances_unknown`**: Figuren ohne klare Zuordnung wirken oft unwichtiger in der Story und sterben dadurch seltener (weil sie kaum erwähnt werden).
        
        **Hinweis**: Feature-Wichtigkeit im Random Forest zeigt *statistische Bedeutung*, nicht *kausalen Einfluss*. 
        """)

    elif title == "Confusion Matrix":
        st.markdown("""
        ### Verteilung der Modell-Fehler und -Treffer (Confusion Matrix)

        Die Matrix zeigt, wie gut das Modell zwischen „Tot“ und „Lebendig“ unterscheidet:

        - **True Negative (oben links)**: 42 Figuren wurden korrekt als „tot“ vorhergesagt.
        - **True Positive (unten rechts)**: 203 Figuren wurden korrekt als „lebendig“ erkannt.
        - **False Positive (oben rechts)**: 27 Figuren wurden fälschlich als „lebendig“ klassifiziert – das Modell unterschätzt hier das Risiko.
        - **False Negative (unten links)**: 40 Figuren wurden fälschlich als „tot“ eingestuft – das Modell ist hier zu pessimistisch.

        #### Interpretation:
        - Das Modell **erkennt Überlebende recht zuverlässig** (hohe True-Positive-Zahl).
        - Die Zahl der **False Negatives ist relativ hoch**. Das Modell „tötet“ also häufiger Charaktere, die eigentlich überleben.
        - Das könnte an **Verzerrungen in den Daten** liegen (z. B. Figuren mit wenig Infos wirken „entbehrlich“).
                    
        **Fazit**: 
                    
        Das Modell ist deutlich besser bei Überlebenden. Es erkennt lebdige Figuren recht gut, hat aber Schwierigkeiten, Tote korrekt zu klassifizieren (Recall für Tote ist schlechter).
        Es vertraut zu stark auf „lebendig“, und macht bei toten Charakteren häufiger Fehler (→ Confusion Matrix zeigt viele False Negatives).
        """)


    elif title == "ROC-Kurve":
        st.markdown("""
        ### ROC-Kurve – Modellgüte bewerten

        Die **ROC-Kurve (Receiver Operating Characteristic)** bewertet die Fähigkeit des Modells, zwischen „lebendig“ und „tot“ zu unterscheiden – **über alle möglichen Schwellenwerte hinweg**.

        **Achsen:**
        - **x-Achse** = False Positive Rate (fälschlich als lebendig erkannt)
        - **y-Achse** = True Positive Rate (korrekt als lebendig erkannt)

        **AUC (Area Under the Curve):**
        - **0.5** → Modell rät zufällig
        - **1.0** → perfekte Trennung zwischen Klassen
        - **Unser Modell: AUC = 0.81** → solide Trennschärfe

        ---

        ### Kritische Bewertung

        - Die Kurve liegt deutlich **über der Zufallsdiagonalen** – das ist ein gutes Zeichen.
        - Aber: **AUC ignoriert Klassenverteilung** – ein gutes AUC bedeutet nicht, dass beide Klassen (z. B. „tot“) gut erkannt werden.
        - AUC allein zeigt **nicht**, wo das Modell Fehler macht (dafür besser: Confusion Matrix).

        **Warum liegt die AUC nicht bei 0.90+?**

        1. **Unvollständige oder unscharfe Daten:**  
        Viele Nebencharaktere haben lückenhafte Angaben, das erschwert präzise Vorhersagen.

        2. **Korrelation statt Kausalität:**  
        Das Modell erkennt Muster wie „mehr Buchauftritte = lebt länger“, was **statistisch sinnvoll**, aber **inhaltlich fragwürdig** ist.

        3. **Fehlendes Kontextwissen:**  
        Das Modell kennt keine Handlungslogik, Plotstruktur oder Beliebtheit. Entscheidende Einflussfaktoren beim Tod einer Figur.

        **Fazit:**  
        Die ROC-Kurve zeigt, wie gut das Modell **theoretisch trennt**, aber sagt **nichts darüber**, **wo** und **warum** es im Einzelfall scheitert.
        """)



    elif title == "Überlebenswahrscheinlichkeit Histogramm":
        st.markdown("""
        ### Histogramm der Überlebenswahrscheinlichkeiten

        Dieses Histogramm zeigt, **wie sicher sich das Modell bei seinen Vorhersagen fühlt**, also:  
        Wie stark tendiert es zur Aussage „lebt“ oder „stirbt“?

        #### Interpretation der Achsen:
        - **x-Achse**: vorhergesagte Überlebenswahrscheinlichkeit (zwischen 0 = sicher tot und 1 = sicher lebendig)
        - **y-Achse**: Anzahl der Figuren mit dieser Wahrscheinlichkeit

        ---

        ### Was fällt auf?

        - **Hoher Peak nahe 1.0** → Das Modell ist sich bei vielen Charakteren sehr sicher, dass sie **überleben**.
        - **Wenige Vorhersagen bei 0.5** → Das Modell trifft **selten unsichere Aussagen**. Das spricht für gute Trennbarkeit.
        - **Rechtsschiefe Verteilung** → Ein Großteil der Charaktere wird mit hoher Überlebenswahrscheinlichkeit bewertet. Das könnte entweder:
            - ... ein echter Effekt im Datensatz sein (viele überleben), oder
            - ... eine **Modellverzerrung** durch unbalancierte Klassen (→ siehe Klassenverteilung!).

        ---

        ### Kritische Reflexion

        - Das Modell „meidet“ Mittelfälle; das ist gut für Klarheit, aber gefährlich bei Unsicherheit.
        - Kein perfektes Kalibrierungsmaß: Es zeigt, wie das Modell **sich selbst einschätzt**, nicht wie gut diese Einschätzung wirklich ist. Dafür bräuchte man **Calibration Plots**.
        - Wenn fast alle Wahrscheinlichkeiten bei 0 oder 1 liegen, besteht **Gefahr von Overconfidence**.

        **Fazit:**  
        Das Modell trifft klare Aussagen und wirkt optimistisch, aber ohne die Confusion Matrix würde man verkennen, wie oft das Modell bei den Toten danebenliegt.
        """)

    elif title == "Partial Dependence Plot (PDP)":
        st.markdown("""
        ### Partial Dependence Plot (PDP)

        Der **PDP** zeigt, wie sich **ein einzelnes Merkmal** (z. B. das Alter oder die Anzahl toter Verwandter) **auf die Überlebenswahrscheinlichkeit auswirkt – unabhängig vom Rest der Daten**.

        #### Interpretation der Grafik:
        - **Links: Alter**
            - Die Überlebenswahrscheinlichkeit **sinkt mit steigendem Alter**.
            - Besonders deutlich: Figuren über 50 haben deutlich schlechtere Prognosen.
            - **Achtung**: Alterswerte wurden per **Median-Imputation** ergänzt. Die Aussagekraft bei extremen Alterswerten kann verzerrt sein.
        - **Rechts: Anzahl toter Verwandter**
            - Je mehr tote Verwandte, desto **geringer die Überlebenschance**.
            - Mögliche Erklärung: Viele tote Verwandte = gefährliches Umfeld oder Rolle in konfliktreicher Familie.
        
        #### Kritische Einordnung:
        - Der PDP zeigt **durchschnittliche Effekte**, Extremwerte oder Wechselwirkungen mit anderen Features (z. B. "Alter bei Adeligen") werden nicht berücksichtigt.
        - Die Interpretation basiert auf **Modellannahmen**, nicht auf kausalen Zusammenhängen.
        - Gerade bei stark korrelierten Variablen (Alter, Buchauftritte) sollte man **zusätzliche Plots (z. B. SHAP oder ICE)** zur genaueren Analyse verwenden.

        """)


    elif title == "Kumulative Feature-Wichtigkeit":
        st.markdown("""
        ### Kumulative Feature-Wichtigkeit

        Diese Grafik zeigt, wie viel der Modellleistung durch die **wichtigsten Features** erklärt wird, **aufsummiert** von den stärksten bis zu den schwächsten.

        #### Interpretation:
        - Die Kurve steigt **anfangs stark an**: Ein **kleiner Teil der Features (z. B. Top 20)** erklärt bereits den Großteil des Modells.
        - Danach flacht sie ab: Zusätzliche Features tragen **nur noch marginal** zur Modellleistung bei.

        #### Warum ist das wichtig?
        - Diese Analyse hilft zu erkennen, **wie komplex das Modell tatsächlich ist**.
        - Man kann daraus ableiten, ob eine **Reduktion der Features** möglich ist; etwa für ein leichter interpretierbares oder schnelleres Modell.
        
        #### Kritische Bewertung:
        - Diese Darstellung basiert auf der Feature-Wichtigkeit im Random Forest, die **nicht kausal** ist.
        - Feature-Wichtigkeit kann durch **Korrelation** oder **Datendominanz** (z. B. viele Nullwerte bei seltenen Features) verzerrt sein.
        - Besonders bei stark korrelierten Features können **mehrere scheinbar unwichtige Features** gemeinsam Einfluss haben, was hier **nicht sichtbar** ist.

        👉 **Fazit:** Viele Features im Modell sind formal vorhanden, aber nur wenige dominieren die Entscheidungen. Für robuste Interpretationen sollte man Feature-Reduktion testen und mit SHAP-Werten vergleichen.
        """)


    elif title == "Korrelation zwischen Features":
        st.markdown("""
        ### Korrelation zwischen Features

        Die Korrelationsmatrix zeigt, **wie stark zwei Merkmale miteinander zusammenhängen**.

        - **Rot** steht für eine starke **positive Korrelation** (z. B. `book1` und `book2`)
        - **Blau** zeigt eine **negative Korrelation** (z. B. `male` und bestimmte Titel)

        #### Beispiel: Starke Korrelation zwischen `book1` und `book2`

        Figuren, die im ersten Buch auftreten (`book1 = 1`), tauchen **sehr häufig auch im zweiten Buch** auf – das ist logisch (Seriencharaktere), führt aber dazu, dass das Modell diese Features **nicht unabhängig voneinander** bewertet. Sie tragen **redundante Information**.

        #### Was bedeutet das für unser Modell?

        - **Hohe Korrelation = mögliche Redundanz**  
        Wenn zwei Features sehr ähnlich sind, liefern sie dem Modell oft keine zusätzliche Information. Das kann zu **Overfitting** führen.
        
        - **Multikollinearität kann Modellinterpretation verzerren**  
        Bei stark korrelierten Features ist es schwer zu sagen, **welches Merkmal wirklich entscheidend** ist. Feature Importance ist dann ggf. schwer zu deuten.

        - **Einfluss auf PDP & SHAP**  
        Starke Korrelationen können die **Interpretierbarkeit einzelner Merkmale verzerren**, da ein Feature vielleicht nur wichtig erscheint, weil es stark mit einem anderen verbunden ist.

        """)


    elif title == "Beispielbaum aus dem Random Forest":
        st.markdown("""
        ### Beispielbaum aus dem Random Forest

        Diese Visualisierung zeigt **einen einzigen Entscheidungsbaum**, wie er im Random Forest verwendet wird.

        #### Was sieht man hier?
        - **Split-Kriterien**: z. B. `culture_Rivermen <= 0.5`, `book2`, `title_Lady`
        - **Entscheidungslogik**: Welche Bedingungen führen zu einer Vorhersage „lebt“ oder „tot“?
        - **Anzahl Samples** im Knoten und **Verhältnis lebt/tot**
        - **Gini-Wert** als Maß für Reinheit (je kleiner, desto eindeutiger)

        #### Beispielhafte Pfade:
        - **culture_Rivermen = True → lebt mit hoher Wahrscheinlichkeit nicht**
        - **Nicht Rivermen + kein Lady-Titel + Buch2 vorhanden → Überlebenschance hoch**

        #### Kritische Einordnung:
        - Der gezeigte Baum ist **nur ein Beispiel**. Ein Random Forest besteht aus vielen solcher Bäume.
        - Der Baum ist **leicht interpretierbar**, aber **repräsentiert nicht das ganze Modell**.
        - Die Entscheidungslogik kann helfen, **implizite Regeln im Modell zu verstehen**, ist aber **kein Beweis für Kausalität**.
        """)



    elif title == "Charaktertypen (Cluster)":
        st.markdown("""
        ### Charaktertypen (Cluster)

        Hier wurden Figuren anhand ihrer Eigenschaften automatisch zu **Gruppen (Clustern)** zusammengefasst.

        - **Clustering**: Figuren mit ähnlichen Attributen (z. B. Geschlecht, Adel, Haus, Buchauftritte) werden in Gruppen eingeordnet.
        - **Farben**: zeigen unterschiedliche Cluster (z. B. "junge Adelige", "alte Nebenfiguren", "aktive Kämpfer").
        - **Positionen**: sind auf 2 Dimensionen reduziert, basierend auf einer **Hauptkomponentenanalyse (PCA)** oder einer **t-SNE**-Reduktion. Sie geben ein visuelles Gefühl für die Ähnlichkeit der Figuren.

        #### Interpretation:
        - Cluster liegen **nah beieinander**, wenn Figuren **ähnliche Eigenschaften** teilen.
        - **Trennung zwischen Clustern** kann auf klar unterscheidbare Gruppen hinweisen.
        - Die Methode hilft, **verborgene Muster oder Gruppen** im Datensatz zu entdecken.

        #### Kritische Einordnung:
        - Die x- und y-Achsen haben **keine inhaltliche Bedeutung** – sie ergeben sich rein aus der Reduktion.
        - Die Methode ist **explorativ**, d. h. sie zeigt interessante Muster, die aber **nicht automatisch kausale Gruppen** darstellen.
        """)


    elif title == "Klassenverteilung":
        st.markdown("""
        ### Klassenverteilung: Lebendig vs. Tot

        Diese Grafik zeigt, wie **unausgeglichen** die Daten im Trainingsdatensatz sind:

        - Deutlich mehr Figuren sind **am Leben** (`1`) als gestorben (`0`).
        - Das führt zu einer **Klassenungleichheit**, bei der das Modell leicht bevorzugt, die häufigere Klasse („lebt“) vorherzusagen.

        #### Warum ist das problematisch?
        - **Verzerrte Vorhersagen**: Ein Modell kann eine hohe Gesamtgenauigkeit haben, **obwohl** es z. B. fast nie „tot“ vorhersagt.
        - **Vernachlässigte Minderheitsklasse**: Seltenere Klassen (z. B. „tot“) werden schlechter gelernt.

        #### Lösung: SMOTE
        - Mit **SMOTE (Synthetic Minority Over-sampling Technique)** wird die kleinere Klasse synthetisch ergänzt.
        - Ziel: **Balanciertes Lernen**, bessere Erkennung beider Klassen und robustere Vorhersagen.

        """)


    elif title == "Überleben nach Geschlecht":
        st.markdown("""
        ### Überleben nach Geschlecht – Was erkennt das Modell?

        Dieses Balkendiagramm zeigt den Anteil überlebender Figuren nach Geschlecht.
        - **0 = weiblich**, **1 = männlich**
        - Weibliche Charaktere überleben laut Modell **häufiger** als männliche.

        #### Interpretation:
        - Der Unterschied ist statistisch signifikant, aber **nicht extrem**.
        - Das Modell erkennt Muster wie: *"Weibliche Figuren überleben öfter."*

        #### Kritische Bewertung:
        - **Achtung vor Schein-Kausalität**: Das Modell erkennt **Korrelation**, nicht Ursache. 
        - Der Unterschied könnte auf **Storyrollen** beruhen (z. B. Männer kämpfen öfter).
        - **Bias-Gefahr**: Gesellschaftliche Stereotype aus der Story könnten unkritisch übernommen werden.
        - **Modelltransparenz**: Ohne SHAP oder PDP ist unklar, wie **stark** das Merkmal wirklich wirkt.

        Das Feature „Geschlecht“ liefert also Hinweise, sollte aber **nicht isoliert interpretiert** werden.
        """)


    elif title == "Überleben nach Adel":
        st.markdown("""
        ### Überleben nach Adel – Was sagt das Modell?

        Diese Grafik zeigt, ob Figuren mit Adelstitel (isNoble = 1) laut Modell eher überleben als nicht-adelige Figuren (isNoble = 0).

        #### Interpretation:
        - Der Unterschied ist **leicht erkennbar**, aber **nicht sehr stark**.
        - **Nicht-adelige Figuren** scheinen im Schnitt **etwas höhere Überlebensraten** zu haben.

        #### Kritische Bewertung:
        - **Scheinbar paradoxer Effekt**: Adelige werden im Plot oft als zentrale Figuren dargestellt, aber eben auch als Zielscheiben politischer Intrigen.
        - Das Modell erkennt keine "Macht" oder "Plotrelevanz", sondern nur Korrelationen und die deuten hier **keinen klaren Vorteil für Adelige** an.
        - **Feature-Bias möglich**: Vielleicht hängt das Ergebnis mit anderen Merkmalen zusammen (z. B. „Adelige sind häufiger Männer und sterben öfter“).
        - **Fehlende Kausalität**: Nur weil jemand adelig ist, „verursacht“ das kein Überleben oder Sterben.

        Fazit: Der Adelstitel hat **nur begrenzt Aussagekraft** und sollte nie isoliert interpretiert werden.
        """)


    elif title == "Überleben Heirat":
        st.markdown("""
        ### Überleben in Abhängigkeit vom Familienstand

        Diese Grafik zeigt, ob es einen Zusammenhang zwischen dem Merkmal **`isMarried`** und der Überlebenswahrscheinlichkeit von Charakteren gibt.

        #### Interpretation:
        - **Unverheiratete Figuren (0)** haben in diesem Datensatz eine **leicht höhere Überlebensrate**.
        - **Verheiratete Charaktere (1)** überleben statistisch seltener.

        #### Mögliche Erklärungen:
        - **Plot-Mechanik**: Verheiratete Figuren könnten narrativ eher Zielscheibe von Konflikten oder dramatischen Wendungen sein (z. B. politische Ehen, Rachemotive).
        - **Verzerrung durch Nebenfiguren**: Viele unverheiratete Figuren könnten einfache, wenig involvierte Nebenrollen mit geringem Sterberisiko sein.
        - **Kulturelle Muster**: In GoT sind verheiratete Figuren oft in zentrale Familienkonflikte oder Machtspiele verwickelt, das könnte ein Risiko darstellen.

        > Fazit: Die Differenz ist **relativ gering**, aber statistisch erkennbar. Ob Heirat kausal zu höherem Risiko führt, lässt sich **nicht** direkt sagen. Die Grafik zeigt nur einen **Trend**, keine Ursache.
        """)


    elif title == "Überleben nach Alter (Histogramm)":
        st.markdown("""
        ### Altersverteilung bei Überlebenden und Toten

        Diese Visualisierung zeigt, wie sich das **Alter** bei überlebenden und gestorbenen Figuren verteilt. Damit lässt sich z. B. untersuchen, ob **jüngere Charaktere häufiger überleben**.

        **Auffällig ist der starke Peak um 25 Jahre**. Dieser entsteht, weil **fehlende Altersangaben im Datensatz mit dem Median ersetzt** wurden.

        #### Interpretation:

        - Figuren ohne Altersangabe wurden auf den **Medianwert gesetzt**, was zu einer **künstlichen Häufung** in der Mitte führt.
        - Das erschwert die Bewertung echter Zusammenhänge, etwa ob sehr junge oder sehr alte Charaktere bessere Überlebenschancen haben.
        - Trotzdem lässt sich erkennen, dass Figuren **außerhalb des Medianbereichs (besonders ganz jung oder alt)** teils schlechter abschneiden.
        
        > Hinweis: Das Modell kann durch die Median-Füllung gewisse **Alterseffekte unterschätzen oder falsch deuten**.
    """)


    elif title == "Überleben toter Verwandter":
        st.markdown("""
        ### Überleben in Abhängigkeit von toten Verwandten

        Diese Analyse untersucht den Zusammenhang zwischen dem Merkmal **`has_dead_relatives`** und der Überlebensrate von Charakteren.

        #### Interpretation:
        - Charaktere **ohne tote Verwandte** (0) haben eine signifikant **höhere Überlebensrate**.
        - Figuren mit **toten Verwandten** (1) sterben statistisch **häufiger**.

        #### Mögliche Gründe:
        - **Story-Kontext**: Familienkonflikte, Rachegeschichten oder Blutfehden könnten eine höhere Gefahr für Figuren mit toten Verwandten bedeuten.
        - **Datenabhängigkeit**: Das Feature basiert auf vorhandenen Stammbaumdaten. Nebencharaktere ohne dokumentierte Familie erscheinen hier eventuell verzerrt.
        - **Fehlende Tiefe**: Das Feature unterscheidet nicht zwischen Anzahl, Nähe oder Bedeutung der toten Verwandten. Ein toter Cousin zählt genauso wie ein ermordetes Elternteil.

        > Fazit: Das Modell erkennt einen Zusammenhang, aber **ohne inhaltliches Verständnis** für familiäre Beziehungen bleibt es eine **statistische Korrelation ohne Kontext**.
        """)

    elif title == "Einführungskapitel vs. Überleben":
        st.markdown("""
        ### Interpretation: Einführungskapitel vs. Überleben

        Dieser Boxplot zeigt, **in welchem Kapitel** eine Figur eingeführt wurde, getrennt nach Überlebensstatus.

        #### Beobachtungen:
        - **Früh eingeführte Figuren (niedriges Kapitel)** sterben deutlich häufiger.
        - **Spät eingeführte Figuren** haben höhere Überlebenschancen; viele leben sogar bis zum Schluss.
        - Die **Verteilung ist bei Toten breiter**, bei Lebenden kompakter.
        - Es gibt viele **Ausreißer bei den Lebenden**, z. B. Nebenfiguren, die erst spät erscheinen.

        #### Kritische Bewertung:
        - Das Modell nutzt hier ein **Erzählmuster**: Wer früh eingeführt wird, ist öfter Hauptfigur und stirbt eher im Lauf der Handlung.
        - Das ist **kein echter kausaler Zusammenhang**, sondern ein **narrativer Effekt**.
        - Vorsicht: Dieses Feature kann zu **Data Leakage** führen, weil es **implizit den Handlungsverlauf** der Bücher abbildet.
        
        #### Fazit:
        Das Modell erkennt: *„Frühe Einführung = höheres Risiko“*; das ist plausibel, aber kein sachlicher Grund. Deshalb ist **kontextkritische Bewertung wichtig.**
        """)

    elif title == "Überleben nach Haus":
        st.markdown("""
        ### Überlebensraten nach Hauszugehörigkeit

        Dieses Balkendiagramm zeigt, **wie groß der Anteil überlebender Figuren** je nach Haus ist.

        #### Interpretation:
        - **House Frey** und **House Stark** zeigen relativ hohe Überlebensraten – überraschend, da beide Häuser im Plot viele Todesfälle aufweisen.
        - **House Targaryen** hat eine auffällig **niedrige Überlebensrate** – könnte an der hohen Plot-Relevanz und Risikoposition ihrer Mitglieder liegen.
        - **„unknown“** und **„Other“** zeigen hohe Überlebensraten – das sind oft **Nebenfiguren ohne große Handlung**, die schlicht nicht getötet wurden.

        #### Kritische Bewertung:
        - Die Balken zeigen **Mittelwerte mit Unsicherheitsintervallen** – bei kleinen Häusern (z. B. Targaryen) ist die Aussage **weniger stabil**.
        - Die Hauszugehörigkeit ist ein **Proxy-Feature** für narrative Wichtigkeit – es sagt oft mehr über Plotrollen als über reale Überlebensmuster.
        - Auch die Kategorie **„Other“** enthält verschiedene Häuser – Interpretation mit Vorsicht!

        #### Fazit:
        Die Hauszugehörigkeit beeinflusst die Überlebenswahrscheinlichkeit – **aber nicht kausal**, sondern oft **indirekt über narrative Rollen und Screentime**.
        """)

    elif title == "t-SNE Tot vs. Lebendig":
        st.markdown("""
        ### t-SNE Visualisierung: Tot vs. Lebendig

        Diese Darstellung zeigt Figuren in einer **reduzierten 2D-Darstellung** ihrer Eigenschaften (t-SNE). Jede Figur ist ein Punkt, eingefärbt nach Überlebensstatus:
        - **Rot = tot**
        - **Blau = lebt**

        #### Interpretation:
        - Die Punkte gruppieren sich **nach Ähnlichkeiten in den Eingabemerkmalen**. Figuren mit ähnlichem Profil liegen räumlich beieinander.
        - Die Farbverteilung ist **nicht klar getrennt**: Es gibt **viele durchmischte Bereiche**, in denen sowohl Tote als auch Lebende vorkommen.
        - Einige **Cluster sind farblich dominanter** (z. B. rein rot oder rein blau). Dort erkennt das Modell möglicherweise klarere Muster.

        #### Kritische Bewertung:
        - t-SNE zeigt **nur relative Ähnlichkeiten**, aber keine absolute Trennschärfe.
        - Es kann sein, dass Figuren **aus dem gleichen Plotstrang** oder mit ähnlichem Alter/Geschlecht/Haus automatisch in Gruppen landen, ohne dass ein echter Zusammenhang zur Überlebenswahrscheinlichkeit besteht.
        - Dennoch ist es ein **visuelles Indiz**, ob sich Klassen trennen lassen oder **stark überlappen**.
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
