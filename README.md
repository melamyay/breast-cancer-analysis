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
├── pipeline.py          # Script principal (étapes 1 à 10)
├── .gitignore
└── README.md
```

> ⚠️ Les fichiers générés (PNG, CSV intermédiaires) ne sont pas versionnés.  
> ⚠️ `Expression_(Short-read)_Public_26Q1_subsetted.csv` (357 Mo) est exclu du repo.

---

## ⚙️ Installation

```bash
cd BREAST_CANCER
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy scipy scikit-learn matplotlib seaborn
```

---

## Pipeline actuel (`pipeline.py`)

### 1. Chargement
Lecture des fichiers Drug Screen, Subtype Matrix et classification cRegMap.

### 2. Filtrage et nettoyage
Sélection des lignées `BREAST == 1`, suppression des doublons et des colonnes avec > 50% de valeurs manquantes.  
→ **30 lignées | 1360 médicaments**

### 3. PCA
Standardisation + réduction en 2 composantes sur le profil de sensibilité aux médicaments.

### 4. Sous-types moléculaires cRegMap
Classification des lignées en sous-types biologiques via [brcaregmap](https://brcaregmap-781093644550.europe-west1.run.app/) :

| Sous-type | n lignées |
|---|---|
| Luminal | 13 |
| TNBC-Basal | 10 |
| TNBC-Mes | 5 |
| HER2-enriched | 2 |
| Luminal-infiltrated | 0 |

### 5. Fusion Drug Screen + sous-types
Croisement des lignées communes entre le drug screen et la classification cRegMap.  
Sous-types retenus pour les analyses statistiques (n ≥ 3) : **Luminal, TNBC-Basal, TNBC-Mes**.

### 6. Top 10 médicaments par sous-type
Classement des médicaments les plus efficaces (AUC la plus basse) par sous-type.

**Exemples de résultats :**
- TNBC-Basal & TNBC-Mes : TRIPTOLIDE, DOLASTATIN-10, SB-743921
- Luminal : TRIPTOLIDE, ECHINOMYCIN, ROMIDEPSIN
- HER2-enriched : METHOTREXATE, CARFILZOMIB, ELESCLOMOL

### 7. PCA colorée par sous-type
Scatter plot PCA avec couleur par sous-type moléculaire cRegMap.

### 8. Barplots Top 10
Un barplot horizontal par sous-type avec valeurs d'AUC annotées.

### 9. Heatmap
AUC moyennes des meilleurs médicaments par sous-type (palette RdYlGn).

### 10. Tests statistiques
**Kruskal-Wallis** par médicament entre les 3 sous-types retenus.

> ⚠️ Note méthodologique : avec n=30 lignées et 1329 tests, la correction FDR
> (Benjamini-Hochberg) est trop conservative et ne retient aucun médicament.
> On retient un seuil strict **p < 0.01** sur la p-value brute,
> approche justifiée pour les analyses exploratoires sur petites cohortes.

**Résultats :**
- 1329 médicaments testés
- **35 médicaments significatifs** (p < 0.01)
- **68 comparaisons post-hoc** significatives (Mann-Whitney, p < 0.05)

Top médicaments différenciés entre sous-types :
- DECITABINE (p = 0.0004) — agent déméthylant
- ADAVOSERTIB, BERZOSERTIB, VE-821 — inhibiteurs ATR/WEE1 (stress réplicatif, pertinents pour TNBC)
- SCH-900776, LY2603618 — inhibiteurs CHK1

---

## ✅ To-do list

### 🔴 Priorité haute

- [x] Filtrage et nettoyage des lignées breast cancer (DepMap)
- [x] Classification par sous-types moléculaires (cRegMap)
- [x] PCA + visualisations
- [x] Top médicaments par sous-type
- [x] Tests statistiques (Kruskal-Wallis + Mann-Whitney)
- [ ] **Interprétation biologique** des 35 médicaments significatifs

### 🟠 Extensions

- [ ] **Modèle de Machine Learning** — prédire l'AUC à partir des scores TF (`influence_uploadedData.csv`) :
  - Random Forest Regressor
  - ElasticNet
  - Évaluation : R², RMSE, feature importance des régulateurs
- [ ] **Interprétation LLM** — utiliser un LLM pour générer des hypothèses biologiques sur les médicaments identifiés

### 🟢 Bonus

- [ ] **Application Shiny (R)** — interface interactive :
  - Sélecteur de sous-type moléculaire
  - Affichage dynamique du top N médicaments
  - Heatmap interactive (plotly)
  - Boxplots par médicament sélectionné

---

## Données sources

| Fichier | Source | Description |
|---|---|---|
| Drug_sensitivity_AUC | [DepMap](https://depmap.org/portal/download/custom/) | AUC des réponses aux médicaments par lignée |
| Subtype_Matrix | [DepMap](https://depmap.org/portal/download/custom/) | Classification des lignées par type de cancer |
| classification_uploadedData | [brcaregmap](https://brcaregmap-781093644550.europe-west1.run.app/) | Sous-types moléculaires + probabilités |
| influence_uploadedData | [brcaregmap](https://brcaregmap-781093644550.europe-west1.run.app/) | Scores d'influence des régulateurs de transcription |

---

## 👥 Équipe

Projet réalisé en binôme 

AMYAY AMAL — [@melamyay]
COKELAER ALEXIS — [@alexiscokelaer]

