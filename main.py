import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
import shap

# ================================================
# STEP 1: DATEN LADEN UND VORBEREITEN
# ================================================
df = pd.read_csv("public/dataset/got_merged_dataset.csv")
df.columns = df.columns.str.strip()

# =============================
# NEUE FEATURES HINZUFÜGEN
# =============================
df["culture"] = df["culture"].fillna("unknown")
df["house"] = df["house"].fillna("unknown")
df["age"] = df["age"].fillna(df["age"].median())

# Familie – ob überhaupt bekannt
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
df["book_count"] = df[["book1", "book2", "book3", "book4", "book5"]].sum(axis=1)
df["in_all_books"] = (df["book_count"] == 5).astype(int)
df["only_in_one_book"] = (df["book_count"] == 1).astype(int)
df["last_book"] = df[["book1", "book2", "book3", "book4", "book5"]].apply(
    lambda row: max([i+1 for i, val in enumerate(row) if val == 1] or [0]), axis=1
)

# Kombination aus Adelig + verheiratet
df["noble_and_married"] = ((df["isNoble"] == 1) & (df["isMarried"] == 1)).astype(int)

# House gruppieren – sonst zu viele Dummies!
df["house_original"] = df["house"]  # <-- unbedingt vor dem Gruppieren sichern
top_houses = df["house"].value_counts().nlargest(10).index
df["house_grouped"] = df["house"].apply(lambda x: x if x in top_houses else "Other")

# Alter verarbeiten
df["has_age"] = df["age"].notna().astype(int)
df["age_filled"] = df["age"].fillna(df["age"].median())

# Dead Relations (binär + Schwelle)
df["has_dead_relatives"] = (df["numDeadRelations"] > 0).astype(int)
df["many_dead_relatives"] = (df["numDeadRelations"] > df["numDeadRelations"].median()).astype(int)

if "death_year" in df.columns:
    df["is_dead"] = df["death_year"].notna().astype(int)
else:
    print("Spalte 'death_year' nicht vorhanden – Feature 'is_dead' wird übersprungen.")
    df["is_dead"] = 0  # Dummywert, damit Featureliste weiter funktioniert

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


features = [
    "male", "book1", "book2", "book3", "book4", "book5",
    "book_count", "in_all_books", "only_in_one_book", "last_book",
    "isNoble", "isMarried", "noble_and_married",
    "has_mother", "has_father", "has_heir", "has_spouse", "alive_family",
    "is_knight", "is_royalty", "is_maester",
    "has_age", "age_filled",
    "numDeadRelations", "has_dead_relatives", "many_dead_relatives",
    "house_grouped", "culture",
    "is_dead", "introduced_late", "has_allegiances",
    "book_of_death", "book_intro_chapter",
    "a_game_of_thrones", "a_clash_of_kings", "a_storm_of_swords",
    "a_feast_for_crows", "a_dance_with_dragons",
    "allegiance_grouped" 
]


# Nur Zeilen behalten, bei denen Zielwert nicht fehlt
df = df[["S.No", "name"] + features + ["isAlive"]]
df = df[df["isAlive"].notna()]

# One-Hot-Encoding
df_encoded = pd.get_dummies(df.drop(["name", "S.No"], axis=1), columns=["house_grouped", "culture", "allegiance_grouped"])
X = df_encoded.drop("isAlive", axis=1)
y = df_encoded["isAlive"]

# Namen für spätere Zuordnung sichern
names = df["name"]
ids = df["S.No"]

# ================================================
# STEP 2: TRAIN/TEST SPLIT (mit Stratify!)
# ================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("\n NaN-Check:")
print(X.isnull().sum()[X.isnull().sum() > 0])

# ================================================
# STEP 3: OVERSAMPLING MIT SMOTE
# ================================================
print("\n=== Klassenverteilung vor SMOTE ===")
print(y_train.value_counts())

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("\n=== Klassenverteilung nach SMOTE ===")
print(pd.Series(y_train_res).value_counts())

# ================================================
# STEP 4: GRIDSEARCH – BESTES MODELL FINDEN
# ================================================
from sklearn.model_selection import GridSearchCV

print("\n Starte GridSearchCV zur Modelloptimierung ...")

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2"]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(class_weight="balanced", random_state=42),
    param_grid=param_grid,
    scoring="roc_auc",
    cv=5,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train_res, y_train_res)

print("\n Beste Parameterkombination:")
print(grid_search.best_params_)

print("\n Beste AUC-Score:")
print(grid_search.best_score_)

# Bestes Modell speichern
rf = grid_search.best_estimator_

# Modell und Feature-Namen als Pickle-Dateien speichern
import pickle

# Speichern des trainierten Modells
with open("model.pkl", "wb") as f:
    pickle.dump(rf, f)

# Speichern der Feature-Spalten
with open("feature_columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("✅ Modell und Feature-Spalten wurden gespeichert.")

# ================================================
# STEP 5: TESTDATEN BEWERTEN
# ================================================
y_pred = rf.predict(X_test)
probs = rf.predict_proba(X_test)

print("\n=== Modellbewertung ===")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ================================================
# STEP 6: ERGEBNIS-TABELLE SPEICHERN
# ================================================
results = X_test.copy()
results["actual"] = y_test
results["predicted"] = y_pred
results["probability_death"] = probs[:, 0]
results["probability_survival"] = probs[:, 1]
results["name"] = names.loc[X_test.index].values
results["S.No"] = ids.loc[X_test.index].values
results = results[["S.No", "name", "actual", "predicted", "probability_death", "probability_survival"] + list(X_test.columns)]

results.to_csv("public/dataset/got_model_results.csv", index=False)
print("\n Datei 'public/dataset/got_model_results.csv' wurde gespeichert.")

# ================================================
# STEP 7: FEATURE IMPORTANCE VISUALISIERUNG
# ================================================
importances = rf.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df.head(20), x="Importance", y="Feature")
plt.title("Wichtigste Merkmale (Top 20)")
plt.xlabel("Wichtigkeit")
plt.ylabel("Merkmal")
plt.tight_layout()
plt.show()

# ================================================
# STEP 8: CONFUSION MATRIX HEATMAP
# ================================================
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["Tot", "Lebt"], yticklabels=["Tot", "Lebt"])
plt.xlabel("Vorhergesagt")
plt.ylabel("Tatsächlich")
plt.title("Confusion Matrix als Heatmap")
plt.tight_layout()
plt.show()

# ================================================
# STEP 9: ROC-KURVE
# ================================================
fpr, tpr, _ = roc_curve(y_test, probs[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-Kurve – Modellbewertung")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()