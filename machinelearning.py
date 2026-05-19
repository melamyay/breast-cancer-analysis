# =============================================================================
# ML : Prédiction de la réponse aux médicaments et du sous-type moléculaire
#      à partir des scores d'influence des régulateurs de transcription (TF)
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import ElasticNet, LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, classification_report

DATA_DIR = Path("Data")

sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = "Set2"


# =============================================================================
# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
# =============================================================================

# Scores TF : transposition → lignées en lignes, TF en colonnes
influence_df = pd.read_csv(DATA_DIR / "influence_uploadedData.csv", index_col=0)
influence_df = influence_df.T
influence_df.index.name = "Cell_Line"

# Sous-types cRegMap
classif_df = pd.read_csv(DATA_DIR / "classification_uploadedData.csv")
classif_df = classif_df[["Cell.lines", "cluster"]].rename(
    columns={"Cell.lines": "Cell_Line", "cluster": "Subtype"}
)

# Drug screen
drug_df       = pd.read_csv(DATA_DIR / "breast_drug_sensitivity_clean.csv")
cell_line_col = drug_df.columns[0]

# Fusion TF + sous-types + drug screen
tf_subtype = influence_df.merge(classif_df, left_index=True, right_on="Cell_Line")
tf_drug    = influence_df.merge(
    drug_df.set_index(cell_line_col),
    left_index=True, right_index=True
)

print(f"Lignées disponibles pour ML : {tf_subtype.shape[0]}")
print(f"Features TF                : {influence_df.shape[1]}")
print(f"\nDistribution des sous-types :")
print(tf_subtype["Subtype"].value_counts())


# =============================================================================
# 2. PARTIE A – PRÉDICTION DU SOUS-TYPE MOLÉCULAIRE
#    Features : scores TF | Cible : Subtype (Luminal, TNBC-Basal, TNBC-Mes…)
# =============================================================================

print("\n" + "="*60)
print("PARTIE A — Prédiction du sous-type moléculaire")
print("="*60)

X_cls = tf_subtype.drop(columns=["Cell_Line", "Subtype"]).values
y_cls = LabelEncoder().fit_transform(tf_subtype["Subtype"])
label_names = tf_subtype["Subtype"].unique()

scaler_cls = StandardScaler()
X_cls_sc   = scaler_cls.fit_transform(X_cls)

# LOO cross-validation (adapté aux petits échantillons)
loo = LeaveOneOut()

models_cls = {
    "Random Forest":      RandomForestClassifier(n_estimators=100, random_state=42),
    "Régression logist.": LogisticRegression(max_iter=1000, random_state=42),
}

results_cls = {}
for name, model in models_cls.items():
    X_in = X_cls if name == "Random Forest" else X_cls_sc
    scores = cross_val_score(model, X_in, y_cls, cv=loo, scoring="accuracy")
    results_cls[name] = scores.mean()
    print(f"{name:25s} → Accuracy LOO : {scores.mean():.3f}")

# Entraînement final Random Forest pour feature importance
rf_cls = RandomForestClassifier(n_estimators=100, random_state=42)
rf_cls.fit(X_cls, y_cls)

fi_cls = pd.Series(
    rf_cls.feature_importances_,
    index=influence_df.columns
).sort_values(ascending=False).head(20)

# Barplot feature importance
fig, ax = plt.subplots(figsize=(10, 7))
fi_cls.sort_values().plot.barh(ax=ax, color="steelblue", edgecolor="white")
ax.set_title("Top 20 régulateurs prédictifs du sous-type moléculaire\n(Random Forest)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Importance (Gini)", fontsize=11)
plt.tight_layout()
plt.savefig(DATA_DIR / "ml_feature_importance_subtype.png", dpi=300, bbox_inches="tight")
plt.show()

# Barplot comparaison des modèles
fig, ax = plt.subplots(figsize=(7, 4))
colors = sns.color_palette(PALETTE, n_colors=len(results_cls))
ax.barh(list(results_cls.keys()), list(results_cls.values()),
        color=colors, edgecolor="white")
for i, (name, val) in enumerate(results_cls.items()):
    ax.text(val + 0.005, i, f"{val:.3f}", va="center", fontsize=10)
ax.set_xlim(0, 1.1)
ax.set_xlabel("Accuracy (LOO cross-validation)", fontsize=11)
ax.set_title("Comparaison des modèles – Prédiction du sous-type",
             fontsize=12, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(DATA_DIR / "ml_model_comparison_subtype.png", dpi=300, bbox_inches="tight")
plt.show()


# =============================================================================
# PARTIE B – Prédiction de l'AUC par médicament (KFold 5)
# =============================================================================

from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)

kw_df     = pd.read_csv(DATA_DIR / "kruskal_wallis_results.csv")
top5_drugs = kw_df.nsmallest(5, "p_value")["Drug"].tolist()

X_reg    = influence_df.loc[tf_drug.index].values
scaler_r = StandardScaler()
X_reg_sc = scaler_r.fit_transform(X_reg)

models_reg = {
    "Random Forest":   RandomForestRegressor(n_estimators=100, random_state=42),
    "ElasticNet":      ElasticNet(max_iter=5000, random_state=42),
    "Régression lin.": LinearRegression(),
}

all_results = []

for drug in top5_drugs:
    short  = drug.split(" (")[0]
    y_full = tf_drug[drug].values

    # Filtrer les NaN
    mask  = ~np.isnan(y_full)
    X_ok  = X_reg[mask]
    Xs_ok = X_reg_sc[mask]
    y_ok  = y_full[mask]

    print(f"\n{short} — {mask.sum()} lignées disponibles")

    for name, model in models_reg.items():
        X_in = X_ok if name == "Random Forest" else Xs_ok
        r2   = cross_val_score(model, X_in, y_ok, cv=kf, scoring="r2")
        rmse = cross_val_score(model, X_in, y_ok, cv=kf,
                               scoring="neg_mean_squared_error")
        r2_mean   = r2.mean()
        rmse_mean = np.sqrt(-rmse.mean())
        all_results.append({
            "Drug": short, "Modèle": name,
            "R2_CV": r2_mean, "RMSE_CV": rmse_mean
        })
        print(f"  {name:20s} → R² = {r2_mean:.3f} | RMSE = {rmse_mean:.4f}")

results_reg = pd.DataFrame(all_results)
results_reg.to_csv(DATA_DIR / "ml_regression_results.csv", index=False)

# Feature importance RF pour DECITABINE
best_drug  = kw_df.nsmallest(1, "p_value")["Drug"].values[0]
short_best = best_drug.split(" (")[0]
y_best     = tf_drug[best_drug].values
mask_best  = ~np.isnan(y_best)
X_best     = X_reg[mask_best]

rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_best, y_best[mask_best])

fi_reg = pd.Series(rf_reg.feature_importances_,
                   index=influence_df.columns).sort_values(ascending=False).head(20)

fig, ax = plt.subplots(figsize=(10, 7))
fi_reg.sort_values().plot.barh(ax=ax, color="coral", edgecolor="white")
ax.set_title(f"Top 20 régulateurs prédictifs de l'AUC – {short_best}\n(Random Forest)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Importance (MSE)", fontsize=11)
plt.tight_layout()
plt.savefig(DATA_DIR / f"ml_feature_importance_{short_best}.png", dpi=300, bbox_inches="tight")
plt.show()

# Heatmap R²
pivot_r2 = results_reg.pivot(index="Drug", columns="Modèle", values="R2_CV")

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(pivot_r2, annot=True, fmt=".3f", cmap="RdYlGn",
            linewidths=0.4, linecolor="white",
            cbar_kws={"label": "R² (5-fold CV)", "shrink": 0.7},
            vmin=-1, vmax=1, ax=ax)
ax.set_title("R² 5-fold CV par médicament et modèle\n(vert = bonne prédiction)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
plt.tight_layout()
plt.savefig(DATA_DIR / "ml_r2_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

print("\n ML terminé — fichiers sauvegardés dans Data/")