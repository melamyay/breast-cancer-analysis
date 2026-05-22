# =============================================================================
# PIPELINE : Analyse de sensibilité aux médicaments – Cancer du sein (PRISM)
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from pathlib import Path
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

DATA_DIR   = Path("Data")
OUTPUT_DIR = Path("Output")
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = "Set2"


# =============================================================================
# 1. CHARGEMENT
# =============================================================================

drug_df    = pd.read_csv(DATA_DIR / "Drug_sensitivity_AUC_(PRISM_Repurposing_Secondary_Screen)_subsetted.csv")
subtype_df = pd.read_csv(DATA_DIR / "Subtype_Matrix_Public_26Q1_subsetted-2.csv")
classif_df = pd.read_csv(DATA_DIR / "classification_uploadedData.csv")

drug_id_col    = drug_df.columns[0]
subtype_id_col = subtype_df.columns[0]


# =============================================================================
# 2. FILTRAGE ET NETTOYAGE – Lignées cancer du sein
# =============================================================================

breast_ids     = subtype_df.loc[subtype_df["BREAST"] == 1, subtype_id_col]
breast_drug_df = drug_df[drug_df[drug_id_col].isin(breast_ids)]
breast_drug_df = breast_drug_df.drop_duplicates()
breast_drug_df = breast_drug_df.loc[:, breast_drug_df.isna().mean() < 0.5]

print(f"Lignées Breast : {breast_drug_df.shape[0]} | Médicaments : {breast_drug_df.shape[1] - 1}")

breast_drug_df.to_csv(OUTPUT_DIR / "breast_drug_sensitivity_clean.csv", index=False)


# =============================================================================
# 3. PRÉPARATION – PCA sur les données drug screen
# =============================================================================

df            = pd.read_csv(OUTPUT_DIR / "breast_drug_sensitivity_clean.csv")
cell_line_col = df.columns[0]
drug_cols     = [col for col in df.columns if col != cell_line_col]

X          = df.drop(columns=[cell_line_col]).fillna(df.mean(numeric_only=True))
X_scaled   = StandardScaler().fit_transform(X)
pca_result = PCA(n_components=2).fit_transform(X_scaled)


# =============================================================================
# 4. SOUS-TYPES MOLÉCULAIRES cREGMAP
# =============================================================================

classif_df = classif_df[["Cell.lines", "cluster", "maxProb"]].rename(
    columns={"Cell.lines": "Cell_Line", "cluster": "Subtype"}
)

print(f"\nDistribution des sous-types moléculaires (cRegMap, 71 lignées) :")
print(classif_df["Subtype"].value_counts())


# =============================================================================
# 5. FUSION DRUG SCREEN + SOUS-TYPES
# =============================================================================

merged_df     = df.merge(classif_df, left_on=cell_line_col, right_on="Cell_Line")
subtype_means = merged_df.groupby("Subtype")[drug_cols].mean()

# Seuil >= 3 lignées par groupe pour les tests statistiques
# Note : sur les 30 lignées du merge, HER2-enriched n'a que 2 lignées
# (les 8 autres n'ont pas de drug screen dans DepMap) → exclu automatiquement
valid_subtypes = merged_df["Subtype"].value_counts()
valid_subtypes = valid_subtypes[valid_subtypes >= 3].index.tolist()
subtype_ids    = sorted(valid_subtypes)

print(f"\nDistribution dans le jeu d'analyse (30 lignées avec drug screen) :")
print(merged_df["Subtype"].value_counts())
print(f"\nSous-types retenus pour les tests (n >= 3) : {subtype_ids}")

subtype_means.to_csv(OUTPUT_DIR / "subtype_drug_means.csv")


# =============================================================================
# 6. TOP 10 MÉDICAMENTS PAR SOUS-TYPE
# =============================================================================

for subtype in subtype_means.index:
    print(f"\n=== Top 10 – {subtype} ===")
    for drug, auc in subtype_means.loc[subtype].sort_values().head(10).items():
        print(f"  {drug.split(' (')[0]} : AUC = {auc:.4f}")


# =============================================================================
# 7. PCA COLORÉE PAR SOUS-TYPE BIOLOGIQUE
# =============================================================================

pca_df = pd.DataFrame({
    "Cell_Line": df[cell_line_col],
    "PCA1":      pca_result[:, 0],
    "PCA2":      pca_result[:, 1],
}).merge(classif_df, on="Cell_Line")

subtypes = pca_df["Subtype"].unique()
palette  = dict(zip(subtypes, sns.color_palette(PALETTE, n_colors=len(subtypes))))

fig, ax = plt.subplots(figsize=(9, 7))

for subtype, group in pca_df.groupby("Subtype"):
    ax.scatter(group["PCA1"], group["PCA2"],
               label=subtype, color=palette[subtype],
               s=120, edgecolor="white", linewidth=0.6)

ax.set_title("PCA – Lignées de cancer du sein\npar sous-type moléculaire (cRegMap)",
             fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Composante principale 1", fontsize=11)
ax.set_ylabel("Composante principale 2", fontsize=11)
ax.legend(title="Sous-type", frameon=True, framealpha=0.8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pca_subtypes_breast.png", dpi=300)
plt.show()


# =============================================================================
# 8. BARPLOTS – Top 10 médicaments par sous-type
# CORRECTION : colors basé sur subtype_means.index pour générer TOUS les barplots
# =============================================================================

colors = sns.color_palette(PALETTE, n_colors=len(subtype_means.index))

for subtype, color in zip(subtype_means.index, colors):
    top_drugs   = subtype_means.loc[subtype].sort_values().head(10)
    short_names = [d.split(" (")[0] for d in top_drugs.index]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(short_names, top_drugs.values, color=color, edgecolor="white")

    for bar, val in zip(bars, top_drugs.values):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9, color="dimgray")

    ax.set_title(f"Top 10 médicaments – {subtype}",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("AUC moyenne (↓ = plus efficace)", fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, top_drugs.values.max() * 1.15)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"top10_drugs_{subtype.replace(' ', '_')}.png", dpi=300)
    plt.show()


# =============================================================================
# 9. HEATMAP – AUC moyennes des meilleurs médicaments par sous-type
# =============================================================================

top_drugs_all = list({
    drug
    for subtype in subtype_means.index
    for drug in subtype_means.loc[subtype].sort_values().head(10).index
})

heatmap_data         = subtype_means[top_drugs_all].copy()
heatmap_data.columns = [c.split(" (")[0] for c in heatmap_data.columns]

fig, ax = plt.subplots(figsize=(20, max(4, len(subtype_ids) * 1.2)))
sns.heatmap(heatmap_data, cmap="RdYlGn_r", annot=True, fmt=".2f",
            linewidths=0.4, linecolor="white", annot_kws={"size": 8},
            cbar_kws={"label": "AUC moyenne", "shrink": 0.7}, ax=ax)

ax.set_title("Sensibilité moyenne aux principaux médicaments par sous-type moléculaire\n(rouge = AUC basse = plus efficace)",
             fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Médicaments", fontsize=11)
ax.set_ylabel("Sous-type", fontsize=11)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "heatmap_top_drugs_subtypes.png", dpi=300, bbox_inches="tight")
plt.show()


# =============================================================================
# 10. TESTS STATISTIQUES – Kruskal-Wallis + Post-hoc Mann-Whitney
# =============================================================================
# Correction pour tests multiples : FDR Benjamini-Hochberg (q < 0.05)
# 3 sous-types testés : Luminal (n=13), TNBC-Basal (n=10), TNBC-Mes (n=5)
# HER2-enriched exclu : seulement 2 lignées dans le jeu drug screen
# =============================================================================

# --- 10.1 Kruskal-Wallis par médicament ---

kw_results = []

for drug in drug_cols:
    groups = [merged_df.loc[merged_df["Subtype"] == s, drug].dropna().values
              for s in subtype_ids]
    if any(len(g) < 3 for g in groups):
        continue
    try:
        stat, p = stats.kruskal(*groups)
        kw_results.append({"Drug": drug, "KW_stat": stat, "p_value": p})
    except ValueError:
        continue

kw_df = pd.DataFrame(kw_results).dropna(subset=["p_value"])
kw_df = kw_df.sort_values("p_value")

# Correction FDR Benjamini-Hochberg
reject, pvals_corrected, _, _ = multipletests(kw_df["p_value"], method="fdr_bh")
kw_df["p_fdr"]           = pvals_corrected
kw_df["significant_fdr"] = reject

kw_df.to_csv(OUTPUT_DIR / "kruskal_wallis_results.csv", index=False)

sig_drugs = kw_df[kw_df["p_fdr"] < 0.05].copy()
sig_drugs["note"] = "FDR Benjamini-Hochberg q < 0.05"

print(f"\nMédicaments testés : {len(kw_df)}")
print(f"Significatifs sans correction (p < 0.01) : {(kw_df['p_value'] < 0.01).sum()}")
print(f"Significatifs avec FDR BH (q < 0.05)     : {len(sig_drugs)}")

if len(sig_drugs) > 0:
    print("\nMédicaments significatifs après correction FDR :")
    print(sig_drugs[["Drug", "KW_stat", "p_value", "p_fdr"]].to_string(index=False))
else:
    print("\nAucun médicament ne survit à la correction FDR q < 0.05.")
    print("Passage en mode exploratoire : on retient les 10 meilleurs (p brute).")
    sig_drugs = kw_df.head(10).copy()
    sig_drugs["note"] = "Top 10 exploratoire (FDR non significatif)"

sig_drugs.to_csv(OUTPUT_DIR / "significant_drugs.csv", index=False)


# --- 10.2 Post-hoc Mann-Whitney par paire de sous-types ---

posthoc_rows = []

for drug in sig_drugs["Drug"]:
    for s1, s2 in combinations(subtype_ids, 2):
        g1 = merged_df.loc[merged_df["Subtype"] == s1, drug].dropna().values
        g2 = merged_df.loc[merged_df["Subtype"] == s2, drug].dropna().values
        if len(g1) < 3 or len(g2) < 3:
            continue
        stat, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
        posthoc_rows.append({
            "Drug":       drug,
            "Subtype_A":  s1,
            "Subtype_B":  s2,
            "MW_stat":    stat,
            "p_value":    p,
            "mean_AUC_A": g1.mean(),
            "mean_AUC_B": g2.mean(),
        })

posthoc_df = pd.DataFrame(posthoc_rows).sort_values("p_value")
posthoc_df.to_csv(OUTPUT_DIR / "posthoc_mannwhitney_results.csv", index=False)

print(f"\nComparaisons post-hoc significatives (p < 0.05) : {(posthoc_df['p_value'] < 0.05).sum()}")


# --- 10.3 Volcano plot ---

subtype_means_sig = merged_df.groupby("Subtype")[kw_df["Drug"].tolist()].mean()
effect_size       = subtype_means_sig.std(axis=0)

volcano_df                = kw_df.copy()
volcano_df["effect_size"] = volcano_df["Drug"].map(effect_size)
volcano_df["-log10_p"]    = -np.log10(volcano_df["p_value"])
fdr_sig                   = volcano_df[volcano_df["significant_fdr"] == True]

fig, ax = plt.subplots(figsize=(10, 7))

ax.scatter(volcano_df["effect_size"], volcano_df["-log10_p"],
           alpha=0.5, s=60, color="lightsteelblue", edgecolor="white", linewidth=0.4,
           label="Non significatif")
ax.scatter(fdr_sig["effect_size"], fdr_sig["-log10_p"],
           color="crimson", s=90, zorder=5, edgecolor="white", linewidth=0.4,
           label="Significatif (FDR q < 0.05)")

for _, row in fdr_sig.iterrows():
    ax.annotate(row["Drug"].split(" (")[0],
                xy=(row["effect_size"], row["-log10_p"]),
                xytext=(5, 3), textcoords="offset points",
                fontsize=7.5, color="crimson")

ax.axhline(-np.log10(0.01), color="gray", linestyle="--", linewidth=0.8, label="p brute = 0.01")
fdr_threshold = volcano_df.loc[volcano_df["significant_fdr"], "-log10_p"].min() if len(fdr_sig) > 0 else None
if fdr_threshold:
    ax.axhline(fdr_threshold, color="orangered", linestyle=":", linewidth=0.8, label="FDR q = 0.05")

ax.set_xlabel("Dispersion inter-sous-type (std des moyennes AUC)", fontsize=11)
ax.set_ylabel("-log10(p-value)", fontsize=11)
ax.set_title("Médicaments différenciés entre sous-types\n"
             "Kruskal-Wallis + correction FDR Benjamini-Hochberg — n=30 lignées",
             fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "volcano_kruskal_drugs.png", dpi=300, bbox_inches="tight")
plt.show()


# --- 10.4 Boxplots – Top 6 médicaments ---

top6      = kw_df.nsmallest(6, "p_value")["Drug"].tolist()
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes      = axes.flatten()
pal       = sns.color_palette(PALETTE, n_colors=len(subtype_ids))

for ax, drug in zip(axes, top6):
    data_plot = merged_df.loc[merged_df["Subtype"].isin(subtype_ids), ["Subtype", drug]].dropna()

    sns.boxplot(data=data_plot, x="Subtype", y=drug,
                hue="Subtype", palette=pal, width=0.5, legend=False,
                flierprops=dict(marker="o", markersize=4, alpha=0.5), ax=ax)
    sns.stripplot(data=data_plot, x="Subtype", y=drug,
                  color="black", alpha=0.3, size=3, jitter=True, ax=ax)

    p = kw_df.loc[kw_df["Drug"] == drug, "p_value"].values[0]
    ax.set_title(f"{drug.split(' (')[0]}\n(p = {p:.4f})", fontsize=10, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("AUC", fontsize=9)
    ax.tick_params(axis="x", rotation=30)

plt.suptitle("Distribution des AUC pour les 6 médicaments les plus différenciés entre sous-types",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "boxplots_top6_drugs.png", dpi=300, bbox_inches="tight")
plt.show()