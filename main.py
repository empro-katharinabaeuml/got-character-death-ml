import pandas as pd #Tool zum Arbeiten mit Tabellen/Daten
import matplotlib.pyplot as plt #Diagramme zeichnen
import seaborn as sns #hübscheres Tool für Diagramme
from sklearn.ensemble import RandomForestClassifier #Random-Forest-Modell holen
from sklearn.model_selection import train_test_split #Teilt die Daten zufällig in Trainings- und Testdaten auf
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc #Werkzeuge zur Bewertung des Modells
from imblearn.over_sampling import SMOTE #unausgewogene Klassen ausgleichen (z. B. zu viele Überlebende, zu wenige Tote
import shap #Erklärbarkeit des Modells verbessern
import os

def save_plot(filename):
    folder = "public/pictures"
    os.makedirs(folder, exist_ok=True)  # Ordner anlegen, falls nicht vorhanden
    filepath = os.path.join(folder, filename)
    plt.savefig(filepath, bbox_inches='tight')
    print(f"✅ Plot gespeichert unter: {filepath}")

# ================================================
# STEP 1: DATEN LADEN UND VORBEREITEN
# ================================================
df = pd.read_csv("public/dataset/got_merged_dataset.csv") #CSV-Datei mit Daten in Tabelle namens df laden
df.columns = df.columns.str.strip() #Entfernt überflüssige Leerzeichen aus den Spaltennamen

# =============================
# NEUE FEATURES HINZUFÜGEN
# =============================
# Fehlende Werte auffüllen
df["culture"] = df["culture"].fillna("unknown")
df["house"] = df["house"].fillna("unknown")
df["age"] = df["age"].fillna(df["age"].median()) #Wenn age fehlt → der mittlere Wert (Median) aller bekannten Alter

# Familie – ob überhaupt bekannt
#Gibt 1, wenn Mutter/Vater/Erbe/Ehepartner bekannt ist, sonst 0.
df["has_mother"] = df["mother"].notna().astype(int)
df["has_father"] = df["father"].notna().astype(int)
df["has_heir"] = df["heir"].notna().astype(int)
df["has_spouse"] = df["spouse"].notna().astype(int)

# isAlive-Familienfelder auffüllen
df[["isAliveMother", "isAliveFather", "isAliveHeir", "isAliveSpouse"]] = df[
    ["isAliveMother", "isAliveFather", "isAliveHeir", "isAliveSpouse"]
].fillna(0)

# Anzahl lebender Angehöriger
df["alive_family"] = (
    df["isAliveMother"] + df["isAliveFather"] + df["isAliveHeir"] + df["isAliveSpouse"]
).astype(int)


# Titel-Analyse
df["is_knight"] = df["title"].fillna("").str.contains("Knight|Ser", case=False).astype(int)
df["is_royalty"] = df["title"].fillna("").str.contains("King|Queen|Prince|Princess", case=False).astype(int)
df["is_maester"] = df["title"].fillna("").str.contains("Maester", case=False).astype(int)

# Bücher: Letztes Buch, in dem die Figur vorkommt
#Zählt, in wie vielen Büchern die Figur vorkommt
df["book_count"] = df[["book1", "book2", "book3", "book4", "book5"]].sum(axis=1)

#Zählt, in wie vielen Büchern die Figur vorkommt
df["in_all_books"] = (df["book_count"] == 5).astype(int)
df["only_in_one_book"] = (df["book_count"] == 1).astype(int)

#Sucht, in welchem Buch zuletzt die Figur vorkommt.
df["last_book"] = df[["book1", "book2", "book3", "book4", "book5"]].apply(
    lambda row: max([i+1 for i, val in enumerate(row) if val == 1] or [0]), axis=1
)

# Kombination aus Adelig + verheiratet; Gibt 1, wenn jemand adelig und verheiratet ist
df["noble_and_married"] = ((df["isNoble"] == 1) & (df["isMarried"] == 1)).astype(int)

# House gruppieren – sonst zu viele Dummies!
#Nur die Top 10 Häuser bleiben, alle anderen → "Other"
df["house_original"] = df["house"]  # <-- unbedingt vor dem Gruppieren sichern
top_houses = df["house"].value_counts().nlargest(10).index
df["house_grouped"] = df["house"].apply(lambda x: x if x in top_houses else "Other")

# Alter verarbeiten
df["has_age"] = df["age"].notna().astype(int)
df["age_filled"] = df["age"].fillna(df["age"].median())

# Dead Relations (binär + Schwelle)
df["has_dead_relatives"] = (df["numDeadRelations"] > 0).astype(int)
df["many_dead_relatives"] = (df["numDeadRelations"] > df["numDeadRelations"].median()).astype(int)


if "book_intro_chapter" in df.columns:
    df["book_intro_chapter"] = df["book_intro_chapter"].fillna(0)
    df["introduced_late"] = (df["book_intro_chapter"] > 30).astype(int)
else:
    print("Spalte 'book_intro_chapter' nicht vorhanden – Feature 'introduced_late' wird übersprungen.")
    df["book_intro_chapter"] = 0
    df["introduced_late"] = 0

if "allegiances" in df.columns:
    df["has_allegiances"] = df["allegiances"].notna().astype(int)
    df["allegiances"] = df["allegiances"].fillna("unknown")
    top_allegiances = df["allegiances"].value_counts().nlargest(10).index
    df["allegiance_grouped"] = df["allegiances"].apply(lambda x: x if x in top_allegiances else "Other")
else:
    print("Spalte 'allegiances' nicht vorhanden – Features dazu werden übersprungen.")
    df["has_allegiances"] = 0
    df["allegiances"] = "unknown"
    df["allegiance_grouped"] = "Other"


# Fallback für optionale Spalten, falls sie nicht existieren
#Falls wichtige Spalten fehlen → auffüllen
optional_zero_fields = [
    "book_of_death",
    "a_game_of_thrones",
    "a_clash_of_kings",
    "a_storm_of_swords",
    "a_feast_for_crows",
    "a_dance_with_dragons"
]

for col in optional_zero_fields:
    if col not in df.columns:
        print(f"Spalte '{col}' fehlt – wird mit 0 ergänzt.")
        df[col] = 0
    else:
        df[col] = df[col].fillna(0)


# ================================================
# FEATURES DEFINIEREN 
# ================================================
#Auswahl der Spalten (= Features), die für das Modell genutzt werden
features = [
    "title", "male", "culture", "house", "age",
    "book1", "book2", "book3", "book4", "book5",
    "isMarried", "isNoble", "numDeadRelations",
    "allegiances",
    "a_game_of_thrones", "a_clash_of_kings", "a_storm_of_swords",
    "a_feast_for_crows", "a_dance_with_dragons"
]

#Nur Datensätze verwenden, bei denen das Ziel („isAlive“) bekannt ist 
df = df[df["isAlive"].notna()]

# Drop von Leakage-Spalten (falls noch irgendwo vorhanden)
df = df.drop(columns=[
    "book_of_death", "death_year", "death_chapter", "is_dead",
    "isAliveMother", "isAliveFather", "isAliveHeir", "isAliveSpouse",
    "name_y"
], errors="ignore")

# Ziel- und Feature-Set extrahieren
#Ziel- und Feature-Tabelle bauen
df_model = df[["S.No", "name"] + features + ["isAlive"]].copy()

# One-Hot-Encoding für kategoriale Felder
df_model = pd.get_dummies(df_model, columns=["culture", "house", "allegiances", "title"], drop_first=True)

# X und y definieren
# X = alle Merkmale (ohne Seriennummer, Namen, Zielwert)
# y = Zielwert, also isAlive (lebt oder tot)
X = df_model.drop(columns=["S.No", "name", "isAlive"])
y = df_model["isAlive"]

# Namen und IDs speichern (für spätere Rückverknüpfung)
names = df_model["name"]
ids = df_model["S.No"]

# ================================================
# STEP 2: TRAIN/TEST SPLIT (mit Stratify!)
# ================================================
# Trainings- und Testdaten aufteilen
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# NaN-Check (Sicherheitskontrolle); Prüft nochmal, ob irgendwo noch fehlende Werte (NaN) drin sind
print("\n NaN-Check:")
print(X.isnull().sum()[X.isnull().sum() > 0])

# ================================================
# STEP 3: OVERSAMPLING MIT SMOTE
# ================================================
# Zeigt, wie viele Figuren in den Trainingsdaten leben oder tot sind.
print("\n=== Klassenverteilung vor SMOTE ===")
print(y_train.value_counts())

#erstellt künstliche Datenpunkte für die unterrepräsentierte Klasse (hier: „tot“), Ziel: ausgewogenes Verhältnis (Balanced Classes)
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Zeigt, ob das Verhältnis jetzt ausgeglichen ist
print("\n=== Klassenverteilung nach SMOTE ===")
print(pd.Series(y_train_res).value_counts())

# ================================================
# STEP 4: GRIDSEARCH – BESTES MODELL FINDEN
# ================================================
# Tool, um verschiedene Modelleinstellungen systematisch zu testen
from sklearn.model_selection import GridSearchCV

print("\n Starte GridSearchCV zur Modelloptimierung ...")

# Parameter-Raster definieren
param_grid = {
    "n_estimators": [100, 200], #Anzahl der Bäume im Wald
    "max_depth": [None, 10, 20], #maximale Tiefe eines Baumes
    "min_samples_split": [2, 5], #wie viele Datenpunkte mindestens nötig sind, um einen Knoten zu teilen
    "min_samples_leaf": [1, 2], #wie viele Datenpunkte ein Blatt mindestens haben muss
    "max_features": ["sqrt", "log2"] # wie viele Features jeder Baum bei einer Entscheidung in Betracht zieht
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(class_weight="balanced", random_state=42), #das Modell, das optimiert wird (RandomForest)
    param_grid=param_grid, #gleicht Klassenunterschiede aus (Gewichte automatisch setzen)
    scoring="roc_auc", # bewertet die Modelle nach dem AUC-Wert (siehe ROC-Kurve später)
    cv=5, #5-fache Kreuzvalidierung (trainiert 5x mit verschiedenen Splits)
    verbose=1, # zeigt Fortschritt beim Testen
    n_jobs=-1 #nutzt alle CPU-Kerne parallel
)

# Probiert alle Parameterkombis durch und merkt sich das beste Modell
grid_search.fit(X_train_res, y_train_res)

# Bestes Ergebnis anzeigen: beste Parametereinstellung, 
# Den AUC-Score für diese Einstellung, fertige, trainierte Modell rf (RandomForest mit besten Parametern)
rf = grid_search.best_estimator_

print("\n Beste Parameterkombination:")
print(grid_search.best_params_)
print("\n Beste AUC-Score:")
print(grid_search.best_score_)


# Modell und Feature-Spalten speichern
import pickle

with open("model.pkl", "wb") as f:
    pickle.dump(rf, f)

with open("feature_columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)
# Speichert das Modell und die verwendeten Features als .pkl (Pickle-Datei) 

print("Modell und Feature-Spalten wurden gespeichert.")

# ================================================
# STEP 5: TESTDATEN BEWERTEN – MIT SCHWELLEN-ANPASSUNG
# ================================================
# Wahrscheinlichkeiten für die Testdaten berechnen
probs = rf.predict_proba(X_test)

# Schwelle setzen, ab wann jemand als "lebendig" gilt
threshold = 0.65
y_pred = (probs[:, 1] > threshold).astype(int)

print(f"\nBewertung mit Schwelle = {threshold}")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Beste Schwelle automatisch suchen (für Klasse "tot")
from sklearn.metrics import f1_score
import numpy as np

best_thresh = 0.5
best_f1 = 0

for thresh in np.arange(0.4, 0.8, 0.01):
    preds = (probs[:, 1] > thresh).astype(int)
    f1 = f1_score(y_test, preds, pos_label=0)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh

print(f"Beste Schwelle für Klasse 'tot': {best_thresh:.2f} mit F1-Score {best_f1:.3f}")

# ================================================
# STEP 6: ERGEBNIS-TABELLE SPEICHERN
# ================================================
results = X_test.copy() #Test-Ergebnisse in eine Tabelle schreiben

# Tatsächliche und vorhergesagte Werte einfügen
results["actual"] = y_test # Der wahre Zustand (lebt/tot)
results["predicted"] = y_pred # Was das Modell vorausgesagt hat

# Wahrscheinlichkeiten ergänzen
results["probability_death"] = probs[:, 0] #Wahrscheinlichkeit zu sterben
results["probability_survival"] = probs[:, 1] # Wahrscheinlichkeit zu überleben

# Namen & Seriennummer wieder zuordnen
results["name"] = names.loc[X_test.index].values
results["S.No"] = ids.loc[X_test.index].values

#In eine CSV speichern
results.to_csv("public/dataset/got_model_results_clean.csv", index=False)

print(" Ergebnisse gespeichert in 'got_model_results_clean.csv'")

# ================================================
# STEP 7: FEATURE IMPORTANCE VISUALISIERUNG: Welche Merkmale sind wichtig?
# ================================================
# Wichtigkeiten der Merkmale berechnen
importances = rf.feature_importances_
feature_names = X.columns # gibt eine Liste von Zahlen, die zeigen: Wie wichtig jedes Feature für die Entscheidung im Random Forest war

# In DataFrame umwandeln + sortieren
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=False)

# Visualisieren – die Top 20 Features
plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df.head(20), x="Importance", y="Feature")
plt.title("Wichtigste Merkmale (Top 20)")
plt.xlabel("Wichtigkeit")
plt.ylabel("Merkmal")
plt.tight_layout()
save_plot("feature_importance.png")  # Speicherung in public/pictures

# ================================================
# STEP 8: CONFUSION MATRIX HEATMAP: Wie oft lag Modell richtig/falsch?
# ================================================
cm = confusion_matrix(y_test, y_pred) # eine 2×2-Matrix

# Heatmap anzeigen
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["Tot", "Lebt"], yticklabels=["Tot", "Lebt"])
plt.xlabel("Vorhergesagt")
plt.ylabel("Tatsächlich")
plt.title("Confusion Matrix als Heatmap")
plt.tight_layout()
save_plot("heatmap.png")


# ROC-Daten berechnen
fpr, tpr, _ = roc_curve(y_test, probs[:, 1])
roc_auc = auc(fpr, tpr)

# ROC-Kurve zeichnen
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-Kurve – Modellbewertung")
plt.legend(loc="lower right")
plt.tight_layout()
save_plot("roc.png")

from sklearn.decomposition import PCA # Hauptkomponentenanalyse – reduziert die Daten auf wenige Achsen 
from sklearn.cluster import KMeans # Clustering-Methode – gruppiert ähnliche Objekte automatisch

# Datenreduktion auf 2D für die Visualisierung
X_pca = PCA(n_components=2).fit_transform(X)

# Figuren gruppieren mit KMeans
kmeans = KMeans(n_clusters=3).fit(X)

# Zeichnet ein Punktdiagramm
plt.scatter(X_pca[:,0], X_pca[:,1], c=kmeans.labels_, cmap="Set2")
plt.title("Charaktertypen (Cluster)")
plt.tight_layout()
save_plot("cluster.png")

# ================================================
# STEP 9: DECISION TREE: Wie trifft das Modell Entscheidungen? 
# ================================================

from sklearn.tree import export_graphviz
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Einen der Bäume im Random Forest nehmen
tree = rf.estimators_[0]  # z.B. der erste Baum

# Plot mit sklearn
plt.figure(figsize=(20, 10))
plot_tree(tree,
          feature_names=X.columns,
          class_names=["Tot", "Lebt"],
          filled=True,
          max_depth=3,  # Optional: Begrenzung für bessere Übersicht
          fontsize=10)
plt.title("Beispielbaum aus dem Random Forest")
plt.tight_layout()
save_plot("random_forest1.png")


# Feature-Korrelationen
plt.figure(figsize=(12, 10))
sns.heatmap(X.corr(), cmap="coolwarm", center=0)
plt.title("Korrelation zwischen Features")
plt.tight_layout()
save_plot("feature-korrelationen.png")


importance_df["cumulative"] = importance_df["Importance"].cumsum()
plt.plot(range(len(importance_df)), importance_df["cumulative"])
plt.title("Kumulative Feature-Wichtigkeit")
plt.xlabel("Top-N Features")
plt.ylabel("Kumulierte Bedeutung")
plt.grid(True)
plt.tight_layout()
save_plot("kumulative-feature-wichtigkeit.png")


from sklearn.inspection import PartialDependenceDisplay
PartialDependenceDisplay.from_estimator(rf, X, ["age", "numDeadRelations"])
plt.tight_layout()
save_plot("partial_dependence.png")


for i in range(3):
    plt.figure(figsize=(20, 10))
    plot_tree(rf.estimators_[i], feature_names=X.columns, class_names=["Tot", "Lebt"], max_depth=3, filled=True)
    plt.title(f"Baum {i+1}")
    plt.tight_layout()
    save_plot("tree1.png")

sns.histplot(probs[:, 1], bins=30, kde=True)
plt.title("Verteilung der Überlebenswahrscheinlichkeiten")
plt.xlabel("Überlebenswahrscheinlichkeit")
plt.ylabel("Anzahl Figuren")
plt.tight_layout()
save_plot("survival_probability_hist.png")

from sklearn.manifold import TSNE
X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X)
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y, cmap="coolwarm", alpha=0.7)
plt.title("t-SNE Visualisierung: Tot vs. Lebendig")
plt.tight_layout()
save_plot("visualisierung_tot_lebendig.png")

# Klassenverteilung visualisieren; wie viele leben/tot? Dataset unausgewogen?
sns.countplot(x=y)
plt.title("Verteilung: Überlebt vs. Gestorben")
plt.xlabel("isAlive (1=lebt)")
plt.ylabel("Anzahl Charaktere")
plt.tight_layout()
save_plot("klassenverteilung.png")

# Überlebensrate nach Merkmal: Überleben Männer eher als Frauen? Adelige eher als Nicht-Adelige?
sns.barplot(x="male", y="isAlive", data=df)
plt.title("Überlebensrate nach Geschlecht")
plt.ylabel("Anteil lebendig")
plt.tight_layout()
save_plot("survival_by_gender.png")

sns.barplot(x="isNoble", y="isAlive", data=df)
plt.title("Überlebensrate nach Adelig")
plt.ylabel("Anteil adelig")
plt.tight_layout()
save_plot("survival_by_nobility.png")

sns.barplot(x="isMarried", y="isAlive", data=df)
plt.title("Überlebensrate nach Heirat")
plt.ylabel("Anteil verheiratet")
plt.tight_layout()
save_plot("survival_by_isMarried.png")

sns.barplot(x="has_dead_relatives", y="isAlive", data=df)
plt.title("Überlebensrate nach Geschlecht")
plt.ylabel("Anteil hat tote Verwandte")
plt.tight_layout()
save_plot("survival_has_dead_relatives.png")

# Überleben nach Alter: Gibt es eine Altersgruppe, die häufiger stirbt?
sns.histplot(data=df, x="age", hue="isAlive", bins=30, kde=True, multiple="stack")
plt.title("Alter vs. Überlebenswahrscheinlichkeit")
plt.tight_layout()
save_plot("survival_by_age.png")

# Überleben nach Haus (nur Top-Häuser): Gibt es Häuser, bei denen viele sterben?
top_houses = df["house_grouped"].value_counts().nlargest(6).index
sns.barplot(data=df[df["house_grouped"].isin(top_houses)],
            x="house_grouped", y="isAlive")
plt.title("Überleben pro Haus")
plt.xlabel("Haus")
plt.ylabel("Anteil überlebt")
plt.tight_layout()
save_plot("ueberleben_nach_haus.png")

# Buch-Einführung vs. Überleben: Wer spät eingeführt wurde – stirbt er eher?
sns.boxplot(x="isAlive", y="book_intro_chapter", data=df)
plt.title("Einführungskapitel vs. Überleben")
plt.tight_layout()
save_plot("bucheinfuehrung.png")