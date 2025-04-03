# main.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ================================================
# STEP 1: TRAININGSDATEN LADEN UND VORBEREITEN
# ================================================
df_train = pd.read_csv("game_of_thrones_train.csv")
df_train.columns = df_train.columns.str.strip()

# 🔁 Fehlende culture-Werte füllen
df_train["culture"] = df_train["culture"].fillna("unknown")
df_train["house"] = df_train["house"].fillna("unknown")

# Wichtige Features
features = [
    "house", "culture", "male",
    "book1", "book2", "book3", "book4", "book5",
    "popularity", "isNoble", "isMarried"
]

# Nur diese Spalten + Zielvariable behalten
df_train = df_train[features + ["isAlive"]].dropna()

# One-Hot-Encoding für kategorische Merkmale
df_train_encoded = pd.get_dummies(df_train, columns=["house", "culture"])

# Features und Zielvariable definieren
X_train = df_train_encoded.drop("isAlive", axis=1)
y_train = df_train_encoded["isAlive"]

# ================================================
# STEP 2: MODELL TRAINIEREN
# ================================================
print("\n=== Random Forest Training ===")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# ================================================
# STEP 3: TESTDATEN LADEN UND VORBEREITEN
# ================================================
df_test = pd.read_csv("game_of_thrones_test.csv")
df_test.columns = df_test.columns.str.strip()
df_test_original = df_test.copy()

df_test["culture"] = df_test["culture"].fillna("unknown")
df_test["house"] = df_test["house"].fillna("unknown")

df_test = df_test[features].dropna()

# One-Hot-Encoding für Testdaten
df_test_encoded = pd.get_dummies(df_test, columns=["house", "culture"])

# ⚠️ Sicherstellen, dass Testdaten die gleichen Spalten haben wie Training
missing_cols = set(X_train.columns) - set(df_test_encoded.columns)
for col in missing_cols:
    df_test_encoded[col] = 0
df_test_encoded = df_test_encoded[X_train.columns]

X_test = df_test_encoded

# ================================================
# STEP 4: VORHERSAGE AUF TESTDATEN
# ================================================
y_pred_rf = rf.predict(X_test)
probs = rf.predict_proba(X_test)

# Ergebnisse an Originaldaten anhängen
df_test_original = df_test_original.loc[X_test.index].copy()
df_test_original["predicted"] = y_pred_rf
df_test_original["probability_death"] = probs[:, 0]
df_test_original["probability_survival"] = probs[:, 1]

# ================================================
# STEP 5: EXPORT
# ================================================
df_test_original.to_csv("got_test_predictions.csv", index=False)
print("\n📁 Datei 'got_test_predictions.csv' wurde gespeichert.")

# ================================================
# STEP 6: OPTIONAL – VORSCHAU
# ================================================
print("\n=== Vorhersagen (Testdaten) ===")
if "name" in df_test_original.columns:
    print(df_test_original[["name", "predicted", "probability_death", "probability_survival"]].head(10))
else:
    print(df_test_original[["predicted", "probability_death", "probability_survival"]].head(10))
