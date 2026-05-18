# =============================================================================
# PIPELINE : Analyse de sensibilité aux médicaments – Cancer du sein (PRISM)
# =============================================================================

import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = Path("data")


# =============================================================================
# 1. CHARGEMENT DES DONNÉES
# =============================================================================

drug_df    = pd.read_csv(DATA_DIR / "Drug_sensitivity_AUC_(PRISM_Repurposing_Secondary_Screen)_subsetted.csv")
subtype_df = pd.read_csv(DATA_DIR / "Subtype_Matrix_Public_26Q1_subsetted-2.csv")

drug_id_col    = drug_df.columns[0]
subtype_id_col = subtype_df.columns[0]


# =============================================================================
# 2. FILTRAGE : lignées cancer du sein
# =============================================================================

breast_ids      = subtype_df.loc[subtype_df["BREAST"] == 1, subtype_id_col]
breast_drug_df  = drug_df[drug_df[drug_id_col].isin(breast_ids)]

print(f"Lignées Breast dans Drug Screen : {breast_drug_df.shape[0]}")
print(f"Médicaments                     : {breast_drug_df.shape[1] - 1}")

# Nettoyage : doublons + colonnes trop vides (> 50% NaN)
breast_drug_df = breast_drug_df.drop_duplicates()
breast_drug_df = breast_drug_df.loc[:, breast_drug_df.isna().mean() < 0.5]

breast_drug_df.to_csv(DATA_DIR / "breast_drug_sensitivity_clean.csv", index=False)


# =============================================================================
# 3. PCA + CLUSTERING K-MEANS
# =============================================================================

df           = pd.read_csv(DATA_DIR / "breast_drug_sensitivity_clean.csv")
cell_line_col = df.columns[0]

X        = df.drop(columns=[cell_line_col]).fillna(df.mean(numeric_only=True))
X_scaled = StandardScaler().fit_transform(X)

pca_result = PCA(n_components=2).fit_transform(X_scaled)
clusters   = KMeans(n_clusters=3, random_state=42).fit_predict(X_scaled)

cluster_df = pd.DataFrame({
    "Cell_Line": df[cell_line_col],
    "PCA1":      pca_result[:, 0],
    "PCA2":      pca_result[:, 1],
    "Cluster":   clusters
})

cluster_df.to_csv(DATA_DIR / "breast_clusters.csv", index=False)


# =============================================================================
# 4. MOYENNE AUC PAR CLUSTER
# =============================================================================

drug_cols     = [col for col in df.columns if col != cell_line_col]
merged_df     = df.merge(cluster_df, left_on=cell_line_col, right_on="Cell_Line")
cluster_means = merged_df.groupby("Cluster")[drug_cols].mean()

cluster_means.to_csv(DATA_DIR / "cluster_drug_means.csv")


# =============================================================================
# 5. TOP 10 MÉDICAMENTS PAR CLUSTER
# =============================================================================

for cluster in cluster_means.index:
    print(f"\n=== Top 10 médicaments les plus efficaces – Cluster {cluster} ===")
    top_drugs = cluster_means.loc[cluster].sort_values().head(10)
    for drug, auc in top_drugs.items():
        print(f"  {drug} : AUC = {auc:.4f}")


# =============================================================================
# 6. VISUALISATION PCA
# =============================================================================

# Style global
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = "Set2"

# =============================================================================
# 6. PCA – Scatter plot
# =============================================================================

fig, ax = plt.subplots(figsize=(9, 7))

sns.scatterplot(
    data=cluster_df,
    x="PCA1", y="PCA2",
    hue="Cluster",
    palette=PALETTE,
    s=120, edgecolor="white", linewidth=0.6,
    ax=ax
)

# Centroïdes
centroids = cluster_df.groupby("Cluster")[["PCA1", "PCA2"]].mean()
ax.scatter(
    centroids["PCA1"], centroids["PCA2"],
    marker="X", s=250, color="black", zorder=5, label="Centroïde"
)

ax.set_title(
    "PCA – Lignées de cancer du sein\npar profil de sensibilité aux médicaments",
    fontsize=14, fontweight="bold", pad=15
)
ax.set_xlabel("Composante principale 1", fontsize=11)
ax.set_ylabel("Composante principale 2", fontsize=11)
ax.legend(title="Cluster", frameon=True, framealpha=0.8)

plt.tight_layout()
plt.savefig(DATA_DIR / "pca_clusters_breast.png", dpi=300)
plt.show()

# =============================================================================
# 7. BARPLOTS – Top 10 médicaments par cluster
# =============================================================================

colors = sns.color_palette(PALETTE, n_colors=len(cluster_means.index))

for cluster, color in zip(cluster_means.index, colors):
    top_drugs = cluster_means.loc[cluster].sort_values().head(10)
    short_names = [d.split(" (")[0] for d in top_drugs.index]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(
        short_names, top_drugs.values,
        color=color, edgecolor="white", linewidth=0.5
    )

    # Valeurs en bout de barre
    for bar, val in zip(bars, top_drugs.values):
        ax.text(
            val + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=9, color="dimgray"
        )

    ax.set_title(
        f"Top 10 médicaments les plus efficaces – Cluster {cluster}",
        fontsize=13, fontweight="bold", pad=12
    )
    ax.set_xlabel("AUC moyenne (↓ = plus efficace)", fontsize=11)
    ax.set_ylabel("")
    ax.invert_yaxis()
    ax.set_xlim(0, top_drugs.values.max() * 1.15)

    plt.tight_layout()
    plt.savefig(DATA_DIR / f"top10_drugs_cluster_{cluster}.png", dpi=300)
    plt.show()

# =============================================================================
# 8. HEATMAP – AUC moyennes des meilleurs médicaments par cluster
# =============================================================================

# Union des top 10 par cluster (noms raccourcis)
top_drugs_all = list({
    drug
    for cluster in cluster_means.index
    for drug in cluster_means.loc[cluster].sort_values().head(10).index
})

heatmap_data = cluster_means[top_drugs_all].copy()
heatmap_data.columns = [c.split(" (")[0] for c in heatmap_data.columns]

fig, ax = plt.subplots(figsize=(20, 4.5))

sns.heatmap(
    heatmap_data,
    cmap="RdYlGn_r",  # Rouge = AUC basse = efficace
    annot=True,
    fmt=".2f",
    linewidths=0.4,
    linecolor="white",
    annot_kws={"size": 8},
    cbar_kws={"label": "AUC moyenne", "shrink": 0.7},
    ax=ax
)

ax.set_title(
    "Sensibilité moyenne aux principaux médicaments par cluster pharmacologique\n"
    "(rouge = AUC basse = plus efficace)",
    fontsize=14, fontweight="bold", pad=15
)
ax.set_xlabel("Médicaments", fontsize=11)
ax.set_ylabel("Cluster", fontsize=11)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=9)

plt.tight_layout()
plt.savefig(DATA_DIR / "heatmap_top_drugs_clusters.png", dpi=300, bbox_inches="tight")
plt.show()


# =============================================================================
# 9. TESTS STATISTIQUES – Comparaison AUC entre clusters
# =============================================================================
# Objectif : identifier les médicaments pour lesquels les différences d'AUC
# entre clusters sont statistiquement significatives.
#
# Approche :
#   1. Kruskal-Wallis (non-paramétrique, 3 groupes)  → p-value globale
#   2. Correction FDR (Benjamini-Hochberg)            → q-value
#   3. Post-hoc Mann-Whitney par paire de clusters    → médicaments spécifiques
# =============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = Path("data")

df         = pd.read_csv(DATA_DIR / "breast_drug_sensitivity_clean.csv")
cluster_df = pd.read_csv(DATA_DIR / "breast_clusters.csv")

cell_line_col = df.columns[0]
drug_cols     = [col for col in df.columns if col != cell_line_col]

# Fusion avec les clusters
merged_df = df.merge(cluster_df[["Cell_Line", "Cluster"]],
                     left_on=cell_line_col, right_on="Cell_Line")


# -----------------------------------------------------------------------------
# 9.1  Kruskal-Wallis + correction FDR
# -----------------------------------------------------------------------------
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertias = []
K_range = range(2, 8)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.plot(K_range, inertias, marker="o")
plt.xlabel("Nombre de clusters")
plt.ylabel("Inertie")
plt.title("Méthode du coude")
plt.tight_layout()
plt.savefig(DATA_DIR / "elbow_plot.png", dpi=300)
plt.show()

kw_results = []

for drug in drug_cols:
    groups = [
        merged_df.loc[merged_df["Cluster"] == c, drug].dropna().values
        for c in sorted(merged_df["Cluster"].unique())
    ]
    # On ignore les médicaments avec trop peu de valeurs dans un groupe
    if any(len(g) < 3 for g in groups):
        continue

    stat, p = stats.kruskal(*groups)
    kw_results.append({"Drug": drug, "KW_stat": stat, "p_value": p})

kw_df = pd.DataFrame(kw_results)

print(f"Nombre de médicaments testés : {len(kw_results)}")
print(f"Taille des clusters : {merged_df['Cluster'].value_counts()}")

# Correction FDR (Benjamini-Hochberg)
_, q_values, _, _ = multipletests(kw_df["p_value"], method="fdr_bh")
kw_df["q_value"] = q_values

kw_df = kw_df.sort_values("q_value")
kw_df.to_csv(DATA_DIR / "kruskal_wallis_results.csv", index=False)

sig_drugs = kw_df[kw_df["q_value"] < 0.05]
print(f"Médicaments significativement différents entre clusters (q < 0.05) : {len(sig_drugs)}")
print(sig_drugs.head(20).to_string(index=False))

print(f"Nombre de médicaments testés : {len(kw_results)}")
print(f"Taille des clusters : {merged_df['Cluster'].value_counts()}")

# -----------------------------------------------------------------------------
# 9.2  Post-hoc Mann-Whitney par paire de clusters
#      (uniquement sur les médicaments significatifs)
# -----------------------------------------------------------------------------

cluster_ids   = sorted(merged_df["Cluster"].unique())
cluster_pairs = list(combinations(cluster_ids, 2))
posthoc_rows  = []

for drug in sig_drugs["Drug"]:
    for c1, c2 in cluster_pairs:
        g1 = merged_df.loc[merged_df["Cluster"] == c1, drug].dropna().values
        g2 = merged_df.loc[merged_df["Cluster"] == c2, drug].dropna().values

        if len(g1) < 3 or len(g2) < 3:
            continue

        stat, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
        posthoc_rows.append({
            "Drug":      drug,
            "Cluster_A": c1,
            "Cluster_B": c2,
            "MW_stat":   stat,
            "p_value":   p,
            "mean_AUC_A": g1.mean(),
            "mean_AUC_B": g2.mean(),
        })

posthoc_df = pd.DataFrame(posthoc_rows)

# Correction FDR sur les tests post-hoc
_, q_ph, _, _ = multipletests(posthoc_df["p_value"], method="fdr_bh")
posthoc_df["q_value"] = q_ph

posthoc_df = posthoc_df.sort_values("q_value")
posthoc_df.to_csv(DATA_DIR / "posthoc_mannwhitney_results.csv", index=False)

print(f"\nComparaisons post-hoc significatives (q < 0.05) : {(posthoc_df['q_value'] < 0.05).sum()}")


# -----------------------------------------------------------------------------
# 9.3  Visualisation – Volcano plot (KW : effect size vs significance)
# -----------------------------------------------------------------------------

# Effect size approximé : écart-type des moyennes par cluster
cluster_means_sig = merged_df.groupby("Cluster")[sig_drugs["Drug"].tolist()].mean()
effect_size = cluster_means_sig.std(axis=0)   # dispersion inter-cluster

volcano_df = kw_df[kw_df["q_value"] < 0.05].copy()
volcano_df["effect_size"] = volcano_df["Drug"].map(effect_size)
volcano_df["-log10_q"]    = -np.log10(volcano_df["q_value"])

# Labelliser les top 15 les plus significatifs
top_labels = volcano_df.nsmallest(15, "q_value")

fig, ax = plt.subplots(figsize=(10, 7))

ax.scatter(
    volcano_df["effect_size"],
    volcano_df["-log10_q"],
    alpha=0.6, s=60, color="steelblue", edgecolor="white", linewidth=0.4
)
ax.scatter(
    top_labels["effect_size"],
    top_labels["-log10_q"],
    color="crimson", s=80, zorder=5, edgecolor="white", linewidth=0.4
)

for _, row in top_labels.iterrows():
    ax.annotate(
        row["Drug"].split(" (")[0],
        xy=(row["effect_size"], row["-log10_q"]),
        xytext=(5, 3), textcoords="offset points",
        fontsize=7.5, color="crimson"
    )

ax.axhline(-np.log10(0.05), color="gray", linestyle="--", linewidth=0.8,
           label="q = 0.05")

ax.set_xlabel("Dispersion inter-cluster (std des moyennes AUC)", fontsize=11)
ax.set_ylabel("-log10(q-value)", fontsize=11)
ax.set_title(
    "Médicaments significativement différenciés entre clusters\n(Kruskal-Wallis + FDR Benjamini-Hochberg)",
    fontsize=13, fontweight="bold", pad=12
)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(DATA_DIR / "volcano_kruskal_drugs.png", dpi=300, bbox_inches="tight")
plt.show()


# -----------------------------------------------------------------------------
# 9.4  Boxplots – Top 6 médicaments les plus significatifs
# -----------------------------------------------------------------------------

top6 = kw_df.nsmallest(6, "q_value")["Drug"].tolist()
short = lambda d: d.split(" (")[0]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

palette = sns.color_palette("Set2", n_colors=len(cluster_ids))

for ax, drug in zip(axes, top6):
    data_plot = merged_df[["Cluster", drug]].dropna()

    sns.boxplot(
        data=data_plot, x="Cluster", y=drug,
        palette=palette, width=0.5,
        flierprops=dict(marker="o", markersize=4, alpha=0.5),
        ax=ax
    )
    sns.stripplot(
        data=data_plot, x="Cluster", y=drug,
        color="black", alpha=0.3, size=3, jitter=True, ax=ax
    )

    q = kw_df.loc[kw_df["Drug"] == drug, "q_value"].values[0]
    ax.set_title(f"{short(drug)}\n(q = {q:.2e})", fontsize=10, fontweight="bold")
    ax.set_xlabel("Cluster", fontsize=9)
    ax.set_ylabel("AUC", fontsize=9)

plt.suptitle(
    "Distribution des AUC pour les 6 médicaments les plus différenciés entre clusters",
    fontsize=13, fontweight="bold", y=1.01
)
plt.tight_layout()
plt.savefig(DATA_DIR / "boxplots_top6_drugs.png", dpi=300, bbox_inches="tight")
plt.show()