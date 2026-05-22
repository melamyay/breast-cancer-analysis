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
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

DATA_DIR   = Path("Data")
OUTPUT_DIR = Path("Output")
OUTPUT_DIR.mkdir(exist_ok=True)

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

# Drug screen — lu depuis Output/ (généré par pipeline.py)
drug_df       = pd.read_csv(OUTPUT_DIR / "breast_drug_sensitivity_clean.csv")
cell_line_col = drug_df.columns[0]

# Fusion TF + sous-types (71 lignées — toutes celles classifiées par cRegMap)
tf_subtype = influence_df.merge(classif_df, left_index=True, right_on="Cell_Line")

# Fusion TF + drug screen (30 lignées — intersection drug screen ∩ cRegMap)
tf_drug = influence_df.merge(
    drug_df.set_index(cell_line_col),
    left_index=True, right_index=True
)

print(f"Lignées pour classification (TF seuls)   : {tf_subtype.shape[0]}")
print(f"Lignées pour régression AUC (TF + drugs) : {tf_drug.shape[0]}")
print(f"Features TF : {influence_df.shape[1]}")
print(f"\nDistribution des sous-types (71 lignées cRegMap) :")
print(tf_subtype["Subtype"].value_counts())


# =============================================================================
# 2. PARTIE A – PRÉDICTION DU SOUS-TYPE MOLÉCULAIRE
#    Features : scores TF (440) | Cible : Subtype
#    Utilise les 71 lignées classifiées par cRegMap — pas besoin du drug screen
# =============================================================================

print("\n" + "="*60)
print("PARTIE A — Prédiction du sous-type moléculaire (71 lignées)")
print("="*60)

X_cls    = tf_subtype.drop(columns=["Cell_Line", "Subtype"]).values
y_cls    = LabelEncoder().fit_transform(tf_subtype["Subtype"])
loo      = LeaveOneOut()

scaler_cls = StandardScaler()
X_cls_sc   = scaler_cls.fit_transform(X_cls)

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

fig, ax = plt.subplots(figsize=(10, 7))
fi_cls.sort_values().plot.barh(ax=ax, color="steelblue", edgecolor="white")
ax.set_title("Top 20 régulateurs prédictifs du sous-type moléculaire\n(Random Forest)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Importance (Gini)", fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ml_feature_importance_subtype.png", dpi=300, bbox_inches="tight")
plt.show()

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
plt.savefig(OUTPUT_DIR / "ml_model_comparison_subtype.png", dpi=300, bbox_inches="tight")
plt.show()


# =============================================================================
# PARTIE B – Prédiction de l'AUC par médicament
#    Features : scores TF (440) | Cible : AUC drogue
#    Utilise les 30 lignées avec TF + drug screen disponibles
#    LOO cross-validation + réduction PCA 15 composantes
# =============================================================================

loo_reg = LeaveOneOut()

kw_df      = pd.read_csv(OUTPUT_DIR / "kruskal_wallis_results.csv")
top5_drugs = kw_df.nsmallest(5, "p_value")["Drug"].tolist()

X_reg    = influence_df.loc[tf_drug.index].values
scaler_r = StandardScaler()
X_reg_sc = scaler_r.fit_transform(X_reg)

N_COMPONENTS = 15

models_reg = {
    "Random Forest":   RandomForestRegressor(n_estimators=100, random_state=42),
    "ElasticNet":      Pipeline([
                           ("pca", PCA(n_components=N_COMPONENTS)),
                           ("model", ElasticNet(max_iter=5000, random_state=42))
                       ]),
    "Régression lin.": Pipeline([
                           ("pca", PCA(n_components=N_COMPONENTS)),
                           ("model", LinearRegression())
                       ]),
}

all_results = []

print("\n" + "="*60)
print("PARTIE B — Prédiction de l'AUC (30 lignées, LOO + PCA 15 comp.)")
print("="*60)

for drug in top5_drugs:
    short  = drug.split(" (")[0]
    y_full = tf_drug[drug].values

    mask  = ~np.isnan(y_full)
    X_ok  = X_reg[mask]
    Xs_ok = X_reg_sc[mask]
    y_ok  = y_full[mask]

    print(f"\n{short} — {mask.sum()} lignées disponibles")

    for name, model in models_reg.items():
        X_in = X_ok if name == "Random Forest" else Xs_ok
        try:
            y_true_all, y_pred_all = [], []
            for train_idx, test_idx in loo_reg.split(X_in):
                model.fit(X_in[train_idx], y_ok[train_idx])
                y_pred_all.append(model.predict(X_in[test_idx])[0])
                y_true_all.append(y_ok[test_idx][0])
            y_true_all = np.array(y_true_all)
            y_pred_all = np.array(y_pred_all)
            r2_mean   = r2_score(y_true_all, y_pred_all)
            rmse_mean = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
        except Exception as e:
            print(f"  {name:20s} → Erreur : {e}")
            r2_mean, rmse_mean = np.nan, np.nan

        all_results.append({
            "Drug": short, "Modèle": name,
            "R2_CV": r2_mean, "RMSE_CV": rmse_mean
        })
        print(f"  {name:20s} → R² = {r2_mean:.3f} | RMSE = {rmse_mean:.4f}")

results_reg = pd.DataFrame(all_results)
results_reg.to_csv(OUTPUT_DIR / "ml_regression_results.csv", index=False)

# Feature importance RF pour le meilleur médicament
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
plt.savefig(OUTPUT_DIR / f"ml_feature_importance_{short_best}.png", dpi=300, bbox_inches="tight")
plt.show()

# Heatmap R²
pivot_r2 = results_reg.pivot(index="Drug", columns="Modèle", values="R2_CV")

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(pivot_r2, annot=True, fmt=".3f", cmap="RdYlGn",
            linewidths=0.4, linecolor="white",
            cbar_kws={"label": "R² (LOO CV)", "shrink": 0.7},
            vmin=-1, vmax=1, ax=ax)
ax.set_title("R² LOO CV par médicament et modèle\n"
             "(vert = bonne prédiction | PCA 15 composantes pour ElasticNet et Régression lin.)",
             fontsize=12, fontweight="bold", pad=12)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ml_r2_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nML terminé — fichiers sauvegardés dans Output/")