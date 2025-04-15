import pandas as pd

# CSV-Dateien laden
df_main = pd.read_csv("public/dataset/game_of_thrones_train.csv")
df_extra = pd.read_csv("public/dataset/game_of_thrones_character_deaths.csv")

# Spaltennamen säubern (optional, aber empfohlen)
df_main.columns = df_main.columns.str.strip()
df_extra.columns = df_extra.columns.str.strip()

# Falls nötig: Namen vereinheitlichen
df_main["name_clean"] = df_main["name"].str.lower().str.strip()
df_extra["name_clean"] = df_extra["name"].str.lower().str.strip()

# Zusammenführen (left join auf name_clean)
df_merged = pd.merge(df_main, df_extra, on="name_clean", how="left")

# Optional: Original-Namensspalten beibehalten oder bereinigen
df_merged.drop(columns=["name_clean"], inplace=True)

# Neue CSV speichern
df_merged.to_csv("got_merged_dataset.csv", index=False)

print("✅ Fertig! Neue Datei: got_merged_dataset.csv")
