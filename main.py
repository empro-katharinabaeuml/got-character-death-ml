# main.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ================================================
# STEP 1: DATEN LADEN UND VORBEREITEN
# ================================================
df = pd.read_csv("game_of_thrones_train.csv")
df.columns = df.columns.str.strip()

df["culture"] = df["culture"].fillna("unknown")
df["house"] = df["house"].fillna("unknown")

features = [
    "house", "culture", "male",
    "book1", "book2", "book3", "book4", "book5",
    "isNoble", "isMarried"
]

df["name"] = df["name"]
df["S.No"] = df["S.No"]
names = df["name"]
ids = df["S.No"]

# Drop Rows mit NaN in Features/Ziel
df = df[["S.No", "name"] + features + ["isAlive"]].dropna()


# One-Hot-Encoding ohne "name"
df_encoded = pd.get_dummies(df.drop(["name", "S.No"], axis=1), columns=["house", "culture"])

X = df_encoded.drop("isAlive", axis=1)
y = df_encoded["isAlive"]


# ================================================
# STEP 2: TRAIN/TEST SPLIT
# ================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================================
# STEP 3: MODELL TRAINIEREN
# ================================================
print("\n=== Random Forest Training ===")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# ================================================
# STEP 4: TESTDATEN BEWERTEN
# ================================================
y_pred = rf.predict(X_test)
probs = rf.predict_proba(X_test)

print("\n=== Modellbewertung ===")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# STEP 5: ERGEBNIS-TABELLE SPEICHERN
results = X_test.copy()
results["actual"] = y_test
results["predicted"] = y_pred
results["probability_death"] = probs[:, 0]
results["probability_survival"] = probs[:, 1]

results["name"] = names.loc[X_test.index].values
results["S.No"] = ids.loc[X_test.index].values

# Optional: Reihenfolge anpassen
results = results[["S.No", "name", "actual", "predicted", "probability_death", "probability_survival"] + list(X_test.columns)]

# Export
results.to_csv("got_model_results.csv", index=False)
print("\n📁 Datei 'got_model_results.csv' wurde gespeichert.")

# ================================================
# STEP 7: FEATURE IMPORTANCE – VISUALISIERUNG
# ================================================
importances = rf.feature_importances_
feature_names = X.columns

# Sortiert nach Wichtigkeit
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=False)

# Visualisierung als Säulendiagramm
plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df.head(20), x="Importance", y="Feature")
plt.title("🧠 Wichtigste Merkmale (Top 20)")
plt.xlabel("Wichtigkeit")
plt.ylabel("Merkmal")
plt.tight_layout()
plt.show()
