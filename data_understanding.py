# data_understanding_plus.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# 1. Daten einlesen & Spalten aufräumen
df = pd.read_csv("public/dataset/got_merged_dataset.csv")
df.columns = df.columns.str.strip()  # entfernt Whitespace in Spaltennamen

print("ERSTE BLICKE AUF DIE DATEN:")
print(df.head())

# 2. Struktur und Datentypen
print("STRUKTUR DER DATEN:")
print(df.info())

# 3. Statistische Übersicht
print("STATISTISCHE ÜBERSICHT:")
print(df.describe(include='all'))  # zeigt auch Kategorie-Spalten

# 4. Fehlende Werte analysieren
print("FEHLENDE WERTE JE SPALTE:")
print(df.isna().sum().sort_values(ascending=False))

# Visualisierung der Missing Values
msno.matrix(df)
plt.title("Missing Value Matrix")
plt.show()

# 5. Zielvariable analysieren
print("Verteilung der Zielvariable (isAlive):")
print(df["isAlive"].value_counts(normalize=True))

sns.countplot(x="isAlive", data=df)
plt.title("Verteilung: Lebendig (1) vs. Tot (0)")
plt.xlabel("isAlive")
plt.ylabel("Anzahl")
plt.show()

# 6. Altersverteilung prüfen
if "age" in df.columns:
    sns.histplot(df["age"].dropna(), bins=30, kde=True)
    plt.title("Altersverteilung der Charaktere")
    plt.xlabel("Alter")
    plt.ylabel("Anzahl")
    plt.show()
else:
    print("Keine 'age'-Spalte gefunden.")

# 7. Popularität vs. Überlebensstatus (Korrelation)
if "popularity" in df.columns:
    sns.boxplot(x="isAlive", y="popularity", data=df)
    plt.title("Beliebtheit nach Überlebensstatus")
    plt.xlabel("isAlive")
    plt.ylabel("Popularity")
    plt.show()

# 8. Korrelationen (nur numerisch)
plt.figure(figsize=(10, 8))
corr_matrix = df.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Korrelationsmatrix")
plt.show()

# 9. Kategorische Spalten analysieren
categorical_cols = df.select_dtypes(include="object").columns
for col in categorical_cols:
    print(f"\n🔠 Kategorie-Spalte: {col}")
    print(df[col].value_counts())
    sns.countplot(y=col, data=df, order=df[col].value_counts().index)
    plt.title(f"Häufigkeit: {col}")
    plt.show()

# 10. Überlebensrate nach Kategorie-Feature
for col in categorical_cols:
    if df[col].nunique() < 25:  # nur übersichtliche Kategorien
        ct = pd.crosstab(df[col], df["isAlive"], normalize="index")
        ct.plot(kind="bar", stacked=True)
        plt.title(f"Überlebensrate nach {col}")
        plt.ylabel("Anteil")
        plt.show()
