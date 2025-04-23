import pandas as pd #Tool zum Arbeiten mit Tabellen/Daten
import matplotlib.pyplot as plt #Diagramme zeichnen
import seaborn as sns #hübscheres Tool für Diagramme
from sklearn.ensemble import RandomForestClassifier #Random-Forest-Modell holen
from sklearn.model_selection import train_test_split #Teilt die Daten zufällig in Trainings- und Testdaten auf
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc #Werkzeuge zur Bewertung des Modells
from imblearn.over_sampling import SMOTE #unausgewogene Klassen ausgleichen (z. B. zu viele Überlebende, zu wenige Tote
import os

from visualizations import generate_all_plots  

# Umschalten: Soll mit oder ohne Buchfeatures gearbeitet werden?
USE_BOOK_FEATURES = True  # oder False

# ================================================
# STEP 1: DATEN LADEN UND VORBEREITEN
# ================================================
df = pd.read_csv("public/dataset/got_merged_dataset.csv") #CSV-Datei mit Daten in Tabelle namens df laden
df.columns = df.columns.str.strip() #Entfernt überflüssige Leerzeichen aus den Spaltennamen

# =============================
# NEUE FEATURES HINZUFÜGEN / NEUE MERKMALE SCHAFFEN
# =============================
# Fehlende Werte auffüllen, weil mit Nan Maschine nicht gut lernen kann
df["culture"] = df["culture"].fillna("unknown")
df["house"] = df["house"].fillna("unknown")
df["age"] = df["age"].fillna(df["age"].median()) #Wenn age fehlt → der mittlere Wert (Median) aller bekannten Alter

# Neue Features aus vorhandenen Spalten ableiten
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

# ================================================
# FEATURES DEFINIEREN 
# ================================================
#Auswahl der Spalten (= Features), die für das Modell genutzt werden
# Features MIT Buchinfos (original)
features_with_books = [
    "title", "male", "culture", "house", "age",
    "book1", "book2", "book3", "book4", "book5",
    "isMarried", "isNoble", "numDeadRelations",
    "allegiances"
]

# Features OHNE Buchinfos
features_without_books = [
    "title", "male", "culture", "house", "age",
    "isMarried", "isNoble", "numDeadRelations",
    "allegiances"
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
# Feature-Set abhängig davon wählen, ob Buchfeatures genutzt werden sollen
features = features_with_books if USE_BOOK_FEATURES else features_without_books

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

print(f" Beste Schwelle für Klasse 'tot': {best_thresh:.2f} mit F1-Score {best_f1:.3f}")

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
results.to_csv("public/dataset/got_model_results_clean.csv", index=False, float_format="%.6f")

print(" Ergebnisse gespeichert in 'got_model_results_clean.csv'")

# Wahrscheinlichkeiten auf allen Daten berechnen
all_probs = rf.predict_proba(X)

# Original-Datenstruktur (S.No, name, isAlive)
df_export = pd.DataFrame({
    "S.No": ids.values,
    "name": names.values,
    "isAlive": y.values,
    "probability_death": all_probs[:, 0],
    "probability_survival": all_probs[:, 1]
})

# Sicherstellen, dass alles zwischen 0 und 1 liegt
assert df_export["probability_death"].between(0, 1).all(), "Wertfehler: prob_death"
assert df_export["probability_survival"].between(0, 1).all(), "Wertfehler: prob_survival"

# Speichern
df_export.to_csv("public/dataset/got_full_scores.csv", index=False, float_format="%.6f")

print("Zweite CSV-Datei mit Wahrscheinlichkeiten gespeichert unter: public/dataset/got_full_scores.csv")

generate_all_plots(df, X, y, rf, X_test, y_test, y_pred, probs)

