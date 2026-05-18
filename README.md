#  Breast Cancer Drug Sensitivity Analysis

Mini-projet Bioinfo/MP — M2  
Analyse pharmacogénomique des lignées de cancer du sein à partir des données DepMap / cRegMap.

---

##  Objectif

Explorer les données de sensibilité aux médicaments (Drug Screen PRISM) pour identifier des traitements ciblés par sous-type moléculaire de cancer du sein.

---

## 📁 Structure du projet

```
BREAST_CANCER/
│
├── Data/
│   ├── Drug_sensitivity_AUC_(PRISM_Repurposing_Secondary_Screen)_subsetted.csv
│   ├── Subtype_Matrix_Public_26Q1_subsetted-2.csv
│   └── Inferred_Molecular_Subtypes_Public_26Q1_subsetted-2.csv
│
├── pipeline.py          # Script principal 
├── .gitignore
└── README.md
```

> ⚠️ Les fichiers générés (clusters, heatmaps, PNG...) ne sont pas versionnés.  
> ⚠️ `Expression_(Short-read)_Public_26Q1_subsetted.csv` (357 Mo) est exclu du repo.

---

## ⚙️ Installation

```bash
cd BREAST_CANCER
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy scipy statsmodels scikit-learn matplotlib seaborn
```

---

## 🔬 Pipeline actuel

### 1. Chargement des données
- Lecture des fichiers Drug Screen et Subtype Matrix depuis DepMap

### 2. Filtrage — Lignées cancer du sein
- Sélection des lignées `BREAST == 1` depuis la Subtype Matrix
- Nettoyage : suppression des doublons, exclusion des colonnes avec > 50% de valeurs manquantes

### 3. PCA + Clustering K-Means
- Standardisation des données (StandardScaler)
- Réduction de dimension (PCA 2 composantes)
- Clustering K-Means (n_clusters à déterminer via méthode du coude)

### 4. Moyenne AUC par cluster
- Calcul des AUC moyennes par médicament pour chaque cluster pharmacologique

### 5. Top 10 médicaments par cluster
- Classement des médicaments les plus efficaces (AUC la plus basse) par cluster

### 6. Visualisation PCA
- Scatter plot PCA coloré par cluster avec centroïdes

### 7. Barplots Top 10
- Un barplot horizontal par cluster avec valeurs annotées

### 8. Heatmap
- AUC moyennes des meilleurs médicaments par cluster (palette RdYlGn)

---

## ✅ To-do list

### 🔴 Priorité haute (attendu par le prof)

- [ ] **Corriger le clustering** — appliquer la méthode du coude pour choisir le bon `n_clusters` (actuellement 28/1/1, le clustering est déséquilibré)
- [ ] **Intégrer les sous-types moléculaires cRegMap** — utiliser les données de [brcaregmap](https://brcaregmap-781093644550.europe-west1.run.app/) pour classer les lignées en sous-types réels (Luminal A/B, HER2, Basal/TNBC) au lieu d'un clustering purement pharmacologique
- [ ] **Tests statistiques** — comparer les AUC entre sous-types :
  - Kruskal-Wallis par médicament
  - Correction FDR (Benjamini-Hochberg)
  - Post-hoc Mann-Whitney par paire de clusters
  - Volcano plot + boxplots top médicaments

### 🟠 Priorité moyenne (extensions proposées)

- [ ] **Modèle de Machine Learning** — prédire la réponse aux médicaments à partir des données d'influence des régulateurs de transcription (TF influence depuis cRegMap) :
  - Régression linéaire / ElasticNet
  - Random Forest Regressor
  - Évaluation : R², RMSE, feature importance
- [ ] **Interprétation LLM** — utiliser un LLM (GPT / Claude) pour interpréter les résultats et générer des hypothèses biologiques sur les médicaments identifiés

### 🟢 Bonus

- [ ] **Application Shiny (R)** — interface interactive pour explorer les résultats :
  - Sélecteur de cluster / sous-type
  - Affichage dynamique du top N médicaments
  - Heatmap interactive (plotly)
  - Boxplots par médicament sélectionné
  - Export des résultats en CSV

---

## 📊 Données sources

| Fichier | Source | Description |
|---|---|---|
| Drug_sensitivity_AUC | [DepMap](https://depmap.org/portal/download/custom/) | AUC des réponses aux médicaments par lignée |
| Subtype_Matrix | [DepMap](https://depmap.org/portal/download/custom/) | Classification des lignées par type de cancer |
| Inferred_Molecular_Subtypes | [DepMap](https://depmap.org/portal/download/custom/) | Sous-types moléculaires inférés |
| cRegMap Breast | [brcaregmap](https://brcaregmap-781093644550.europe-west1.run.app/) | Influence des régulateurs de transcription |

---

## 👥 Équipe

Projet réalisé en binôme 

AMYAY AMAL — [@melamyay]
COKELAER ALEXIS — [@alexiscokelaer]

