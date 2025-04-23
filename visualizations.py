import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.tree import plot_tree
from sklearn.inspection import PartialDependenceDisplay
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE
from scipy.stats import pointbiserialr

plt.style.use("dark_background")
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.facecolor"] = "#2b2929"
plt.rcParams["figure.facecolor"] = "#2b2929"
plt.rcParams["savefig.facecolor"] = "#2b2929"

def save_plot(filename):
    folder = "public/pictures"
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, filename), bbox_inches='tight')
    plt.close()

def plot_feature_importance(rf, X):
    importances = rf.feature_importances_
    importance_df = pd.DataFrame({"Feature": X.columns, "Importance": importances})
    importance_df = importance_df.sort_values("Importance", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df.head(20), x="Importance", y="Feature", color="skyblue")
    plt.title("Wichtigste Merkmale (Top 20)")
    save_plot("feature_importance.png")

    importance_df["cumulative"] = importance_df["Importance"].cumsum()
    plt.plot(range(len(importance_df)), importance_df["cumulative"])
    plt.title("Kumulative Feature-Wichtigkeit")
    plt.xlabel("Top-N Features")
    plt.ylabel("Kumulierte Bedeutung")
    plt.grid(True)
    save_plot("kumulative-feature-wichtigkeit.png")

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["Tot", "Lebt"], yticklabels=["Tot", "Lebt"])
    plt.title("Confusion Matrix als Heatmap")
    save_plot("heatmap.png")

def plot_roc(y_test, probs):
    fpr, tpr, _ = roc_curve(y_test, probs[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title("ROC-Kurve – Modellbewertung")
    plt.legend()
    save_plot("roc.png")

def plot_pca_clusters(X):
    X_pca = PCA(n_components=2).fit_transform(X)
    kmeans = KMeans(n_clusters=3).fit(X)
    plt.scatter(X_pca[:,0], X_pca[:,1], c=kmeans.labels_, cmap="Set2")
    plt.title("Charaktertypen (Cluster)")
    save_plot("cluster.png")

def plot_decision_tree(rf, X):
    plt.figure(figsize=(20, 10))
    plot_tree(rf.estimators_[0], feature_names=X.columns, class_names=["Tot", "Lebt"], filled=True, max_depth=3)
    plt.title("Beispielbaum aus dem Random Forest")
    save_plot("random_forest1.png")

def plot_feature_correlation(X):
    plt.figure(figsize=(12, 10))
    sns.heatmap(X.corr(), cmap="coolwarm", center=0)
    plt.title("Korrelation zwischen Features")
    save_plot("feature-korrelationen.png")

def plot_partial_dependence(rf, X):
    PartialDependenceDisplay.from_estimator(rf, X, ["age", "numDeadRelations"])
    save_plot("partial_dependence.png")

def plot_survival_distribution(probs):
    plt.figure(figsize=(8, 6))
    sns.histplot(probs[:, 1], bins=30, kde=True)
    plt.title("Verteilung der Überlebenswahrscheinlichkeiten")
    save_plot("survival_probability_hist.png")

def plot_tsne(X, y):
    X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X)
    plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y, cmap="coolwarm", alpha=0.7)
    plt.title("t-SNE Visualisierung: Tot vs. Lebendig")
    save_plot("tsne_survival.png")

def plot_class_distribution(y):
    sns.countplot(x=y)
    plt.title("Verteilung: Überlebt vs. Gestorben")
    save_plot("klassenverteilung.png")

def plot_group_survival(df):
    groups = ["male", "isNoble", "isMarried", "has_dead_relatives"]
    for g in groups:
        sns.barplot(x=g, y="isAlive", data=df)
        plt.title(f"Überlebensrate nach {g}")
        save_plot(f"survival_by_{g}.png")
        plt.close()

def plot_age_histogram(df):
    sns.histplot(data=df, x="age", hue="isAlive", bins=30, kde=True, multiple="stack")
    plt.title("Alter vs. Überlebenswahrscheinlichkeit")
    save_plot("survival_by_age.png")

def plot_house_survival(df):
    top_houses = df["house_grouped"].value_counts().nlargest(6).index
    sns.barplot(data=df[df["house_grouped"].isin(top_houses)], x="house_grouped", y="isAlive")
    plt.title("Überleben pro Haus")
    save_plot("survival_by_house.png")

def plot_intro_chapter_survival(df):
    sns.boxplot(x="isAlive", y="book_intro_chapter", data=df)
    plt.title("Einführungskapitel vs. Überleben")
    save_plot("intro_chapter_vs_survival.png")

def plot_calibration(y_test, probs):
    prob_true, prob_pred = calibration_curve(y_test, probs[:,1], n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label='Kalibrierungskurve')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title("Kalibrierung des Modells")
    plt.legend()
    save_plot("calibration_curve.png")

def plot_wrong_predictions(probs, y_test, y_pred):
    wrong_preds = (y_test != y_pred)
    sns.histplot(probs[wrong_preds, 1], bins=20, kde=True)
    plt.title("Wahrscheinlichkeiten bei Fehlklassifikationen")
    save_plot("wrong_prediction_probs.png")

def plot_feature_target_corr(X, y):
    correlations = [pointbiserialr(X[col], y)[0] for col in X.columns]
    corr_df = pd.DataFrame({"Feature": X.columns, "Correlation": correlations})
    corr_df = corr_df.sort_values("Correlation", key=abs, ascending=False)
    sns.barplot(data=corr_df.head(20), x="Correlation", y="Feature")
    plt.title("Korrelation der Features mit Überleben (isAlive)")
    save_plot("feature_target_correlation.png")

def plot_survival_by_book_count(df):
    if "book_count" in df.columns:
        sns.boxplot(x="isAlive", y="book_count", data=df)
        plt.title("Überleben nach Anzahl der Bücher")
        save_plot("survival_by_book_count.png")

def plot_age_vs_prediction_accuracy(df_test, y_test, y_pred):
    df_temp = df_test.copy()
    df_temp["isAlive"] = y_test.values  # Zielwert hinzufügen
    df_temp["predicted"] = y_pred
    df_temp["correct"] = (df_temp["isAlive"] == df_temp["predicted"]).astype(int)

    if "age" in df_temp.columns:
        sns.boxplot(x="correct", y="age", data=df_temp)
        plt.title("Alter bei richtiger vs. falscher Vorhersage")
        save_plot("age_correctness.png")

def plot_survival_by_age_and_books(df):
    if "book_count" in df.columns and "age" in df.columns:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x="age", y="book_count", hue="isAlive", alpha=0.7, palette="coolwarm")
        plt.title("Überleben nach Alter und Buchanzahl")
        save_plot("survival_age_bookcount.png")

def plot_groupwise_survival(df):
    if "male" in df.columns:
        sns.barplot(x="male", y="isAlive", data=df)
        plt.title("Überlebensrate nach Geschlecht")
        plt.ylabel("Anteil lebendig")
        plt.tight_layout()
        save_plot("survival_by_gender.png")
        plt.close()

    if "isNoble" in df.columns:
        sns.barplot(x="isNoble", y="isAlive", data=df)
        plt.title("Überlebensrate nach Adel")
        plt.ylabel("Anteil lebendig")
        plt.tight_layout()
        save_plot("survival_by_nobility.png")
        plt.close()

    if "has_dead_relatives" in df.columns:
        sns.barplot(x="has_dead_relatives", y="isAlive", data=df)
        plt.title("Überlebensrate nach toten Verwandten")
        plt.ylabel("Anteil lebendig")
        plt.tight_layout()
        save_plot("survival_has_dead_relatives.png")
        plt.close()

def plot_survival_by_alive_family(df):
    if "alive_family" in df.columns:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x="alive_family", y="isAlive")
        plt.title("Überlebenswahrscheinlichkeit nach lebender Familie")
        plt.xlabel("Anzahl lebender Angehöriger (alive_family)")
        plt.ylabel("Anteil am Leben")
        plt.tight_layout()
        save_plot("survival_by_alive_family.png")
        plt.close()


def generate_all_plots(df, X, y, rf, X_test, y_test, y_pred, probs):
    print(" Starte Generierung der Visualisierungen...")

    plot_feature_importance(rf, X)
    plot_confusion_matrix(y_test, y_pred)
    plot_roc(y_test, probs)
    plot_pca_clusters(X)
    plot_decision_tree(rf, X)
    plot_feature_correlation(X)
    plot_partial_dependence(rf, X)
    plot_survival_distribution(probs)
    plot_tsne(X, y)
    plot_class_distribution(y)
    plot_group_survival(df)
    plot_age_histogram(df)
    plot_house_survival(df)
    plot_intro_chapter_survival(df)
    plot_calibration(y_test, probs)
    plot_wrong_predictions(probs, y_test, y_pred)
    plot_feature_target_corr(X, y)
    plot_survival_by_book_count(df)
    plot_age_vs_prediction_accuracy(X_test.copy(), y_test, y_pred)
    plot_survival_by_age_and_books(df)
    plot_groupwise_survival(df)
    plot_survival_by_alive_family(df)
    print(" Alle Visualisierungen wurden erstellt und gespeichert.")


