# data_understanding.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# 1. CSV laden & Spalten säubern
# ---------------------------
df = pd.read_csv("game_of_thrones_train.csv")
df.columns = df.columns.str.strip()  # entfernt Leerzeichen aus Spaltennamen

print("\n ERSTE BLICKE AUF DIE DATEN:")
print(df.head())

# ---------------------------
# 2. Struktur & Datentypen
# ---------------------------
print("\n📋 STRUKTUR DER DATEN:")
print(df.info())

# ---------------------------
# 3. Statistische Beschreibung
# ---------------------------
print("\n STATISTISCHE ÜBERSICHT (nur numerisch):")
print(df.describe())

# ---------------------------
# 4. Fehlende Werte auflisten
# ---------------------------
print("\n FEHLENDE WERTE JE SPALTE:")
print(df.isna().sum().sort_values(ascending=False))

# ---------------------------
# 5. Zielvariable prüfen
# ---------------------------
print("\n📈 Verteilung der Zielvariable (isAlive):")
print(df["isAlive"].value_counts())

sns.countplot(x="isAlive", data=df)
plt.title("Verteilung: Lebendig (1) vs. Tot (0)")
plt.xlabel("isAlive")
plt.ylabel("Anzahl")
plt.show()

# ---------------------------
# 6. Altersverteilung
# ---------------------------
if "age" in df.columns:
    sns.histplot(df["age"].dropna(), bins=30)
    plt.title("Verteilung Alter der Charaktere")
    plt.xlabel("Alter")
    plt.ylabel("Anzahl")
    plt.show()
else:
    print(" Keine 'age'-Spalte gefunden.")

# ---------------------------
# 7. Beliebtheit vs. Überleben (optional)
# ---------------------------
if "popularity" in df.columns:
    sns.boxplot(x="isAlive", y="popularity", data=df)
    plt.title("Beliebtheit nach Überlebensstatus")
    plt.xlabel("isAlive")
    plt.ylabel("Popularity")
    plt.show()
