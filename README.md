# Breast Cancer Drug Sensitivity Analysis

Mini-projet Bioinfo/MP — M2  
Analyse pharmacogénomique des lignées de cancer du sein à partir des données DepMap / cRegMap.

---

## Objectif

Explorer les données de sensibilité aux médicaments (Drug Screen PRISM) pour identifier des traitements ciblés par sous-type moléculaire de cancer du sein, en combinant les données DepMap et la classification cRegMap.

---

## Structure du projet

```
BREAST_CANCER/
│
├── Data/
│   ├── Drug_sensitivity_AUC_(PRISM_Repurposing_Secondary_Screen)_subsetted.csv
│   ├── Subtype_Matrix_Public_26Q1_subsetted-2.csv
│   ├── Inferred_Molecular_Subtypes_Public_26Q1_subsetted-2.csv
│   ├── classification_uploadedData.csv      ← sous-types moléculaires (cRegMap)
│   └── influence_uploadedData.csv           ← scores TF (cRegMap)
│
├── pipeline.py          # Analyse pharmacogénomique complète (étapes 1–10)
├── machinelearning.py   # Modèles ML (classification + régression)
├── app.R                # Application Shiny interactive
├── .gitignore
└── README.md
```

> ⚠️ Les fichiers générés (PNG, CSV intermédiaires) ne sont pas versionnés.  
> ⚠️ `Expression_(Short-read)_Public_26Q1_subsetted.csv` (357 Mo) est exclu du repo.

---

## ⚙️ Installation

### Python
```bash
cd BREAST_CANCER
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy scipy scikit-learn matplotlib seaborn statsmodels
```

### R (application Shiny)

```r
install.packages(c("shiny", "shinydashboard", "plotly", "DT", "dplyr", "tidyr", "RColorBrewer", "shinyWidgets"))
```

---

### Lancement

```bash
# 1. Pipeline d'analyse
python3 pipeline.py
 
# 2. Machine Learning
python3 machinelearning.py
 
# 3. Application Shiny
Rscript -e "shiny::runApp('app.R', launch.browser=TRUE)"
```

## Pipeline Python (`pipeline.py`)

| Étape | Description | Résultat |
|---|---|---|
| 1 | Chargement Drug Screen + Subtype Matrix | — |
| 2 | Filtrage lignées BREAST + nettoyage | 30 lignées, 1360 médicaments |
| 3 | PCA sur profil pharmacologique | 2 composantes |
| 4 | Sous-types moléculaires cRegMap | 5 sous-types |
| 5 | Fusion drug screen + sous-types | 30 lignées annotées |
| 6 | Top 10 médicaments par sous-type | — |
| 7 | PCA colorée par sous-type | `pca_subtypes_breast.png` |
| 8 | Barplots Top 10 | `top10_drugs_*.png` |
| 9 | Heatmap AUC moyennes | `heatmap_top_drugs_subtypes.png` |
| 10 | Tests statistiques (Kruskal-Wallis + Mann-Whitney) | 35 médicaments significatifs |

**Sous-types identifiés :**

| Sous-type | n lignées (drug screen) |
|---|---|
| Luminal | 13 |
| TNBC-Basal | 10 |
| TNBC-Mes | 5 |
| HER2-enriched | 2 |
| Luminal-infiltrated | 0 |

**Top médicaments par sous-type :**

- **TNBC-Basal** : TRIPTOLIDE, EXATECAN-MESYLATE, DOLASTATIN-10, MAYTANSINOL-ISOBUTYRATE, ECHINOMYCIN, SN38, SB-743921, GEMCITABINE, ROMIDEPSIN, 10-HYDROXYCAMPTOTHECIN
- **TNBC-Mes** : TRIPTOLIDE, DOLASTATIN-10, SEPANTRONIUM BROMIDE, SB-743921, MAYTANSINOL-ISOBUTYRATE, SN38, ECHINOMYCIN, GEMCITABINE, EXATECAN-MESYLATE, ROMIDEPSIN
- **Luminal** : TRIPTOLIDE (AUC = 0.205), ECHINOMYCIN (0.238), ROMIDEPSIN (0.261), SB-743921 (0.265), EXATECAN-MESYLATE (0.360), DOLASTATIN-10 (0.363), OLIGOMYCIN-A (0.366), BGT226 (0.369), SN38 (0.414), MAYTANSINOL-ISOBUTYRATE (0.415)
- **HER2-enriched** : METHOTREXATE (AUC = 0.360), CARFILZOMIB (0.405), ELESCLOMOL (0.444), VOLASERTIB (0.451), BEZ235 (0.470), DINACICLIB (0.473), PANOBINOSTAT (0.476), ALVOCIDIB (0.477), AUY (0.480), AT13387 (0.487)

**Tests statistiques :**
> Avec n=30 lignées et 1329 tests, la correction FDR est trop conservative.
> On retient un seuil **p < 0.01** sur la p-value brute (Kruskal-Wallis),
> approche justifiée pour les analyses exploratoires sur petites cohortes.

- **35 médicaments significatifs** (p < 0.01)
- **68 comparaisons post-hoc** significatives (Mann-Whitney, p < 0.05)
- Top : DECITABINE (p = 0.0004), ADAVOSERTIB, BERZOSERTIB, VE-821, SCH-900776

---

## Machine Learning (`machinelearning.py`)

**Features :** 440 scores d'influence de régulateurs de transcription (TF) — cRegMap  
**Validation :** LOO (classification) | KFold-5 (régression)

### Partie A — Prédiction du sous-type moléculaire

| Modèle | Accuracy (LOO) |
|---|---|
| Random Forest | 84.5% |
| Régression logistique | **85.9%** |

✅ Les scores TF capturent bien l'identité moléculaire des sous-types.

### Partie B — Prédiction de l'AUC par médicament

| Médicament | ElasticNet | Random Forest | Régression lin. |
|---|---|---|---|
| BAY-11-7085 | -0.098 | -0.005 | -0.056 |
| DECITABINE | 0.095 | 0.299 | 0.093 |
| IDAZOXAN | -0.103 | 0.385 | **0.494** |
| LY2603618 | 0.237 | **0.413** | 0.323 |
| VER-49009 | -0.098 | 0.369 | 0.290 |

> R² faibles attendus : dimensionnalité élevée (440 TF) vs petit effectif (n=30).
> Meilleur signal : **IDAZOXAN** (Régression linéaire, R²=0.494) et **LY2603618** (Random Forest, R²=0.413).

---

## Application Shiny (`app.R`)

Interface interactive dark mode avec 7 onglets :

| Onglet | Contenu |
|---|---|
| Vue d'ensemble | Métriques clés, distribution sous-types, top médicaments significatifs |
| PCA | Scatter plot interactif, filtre par sous-type, labels optionnels |
| Top médicaments | Sélecteur de sous-type, slider N, tableau exportable |
| Heatmap | AUC moyennes, filtres sous-types et N médicaments |
| Boxplots | Recherche par médicament, p-value KW en temps réel |
| Tests stat. | Volcano plot, tableau sig_drugs, comparaisons post-hoc |
| Machine Learning | Heatmap R², interprétation, tableau complet |

**Lancement :**
```r
shiny::runApp("app.R")
```

---

## Données sources

| Fichier | Source | Description |
|---|---|---|
| Drug_sensitivity_AUC | [DepMap](https://depmap.org/portal/download/custom/) | AUC des réponses aux médicaments par lignée |
| Subtype_Matrix | [DepMap](https://depmap.org/portal/download/custom/) | Classification des lignées par type de cancer |
| classification_uploadedData | [brcaregmap](https://brcaregmap-781093644550.europe-west1.run.app/) | Sous-types moléculaires + probabilités |
| influence_uploadedData | [brcaregmap](https://brcaregmap-781093644550.europe-west1.run.app/) | Scores d'influence des 440 régulateurs de transcription |

---

## 👥 Équipe

Projet réalisé en binôme 

- AMYAY AMAL — [@melamyay]
- COKELAER ALEXIS — [@alexiscokelaer]

