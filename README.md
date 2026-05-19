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
├── pipeline.py          # Analyse pharmacogénomique (étapes 1 à 10)
├── machinelearning.py   # Modèles ML (classification + régression)
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

## Pipeline principal (`pipeline.py`)

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
Sous-types retenus pour les analyses statistiques (n ≥ 3) : **Luminal, TNBC-Basal, TNBC-Mes**.

### 6–9. Visualisations
- Top 10 médicaments par sous-type (barplots)
- PCA colorée par sous-type moléculaire
- Heatmap AUC moyennes

**Exemples de résultats :**
- TNBC-Basal & TNBC-Mes : TRIPTOLIDE (AUC = 0.076), DOLASTATIN-10, SB-743921
- Luminal : TRIPTOLIDE (AUC = 0.205), ECHINOMYCIN, ROMIDEPSIN
- HER2-enriched : METHOTREXATE, CARFILZOMIB, ELESCLOMOL

### 10. Tests statistiques
**Kruskal-Wallis** par médicament entre les 3 sous-types retenus.

> ⚠️ Note méthodologique : avec n=30 lignées et 1329 tests, la correction FDR est trop
> conservative. On retient un seuil strict **p < 0.01** sur la p-value brute,
> approche justifiée pour les analyses exploratoires sur petites cohortes.

**Résultats :**
- 1329 médicaments testés → **35 significatifs** (p < 0.01)
- **68 comparaisons post-hoc** significatives (Mann-Whitney, p < 0.05)
- Top médicaments : DECITABINE (p = 0.0004), ADAVOSERTIB, BERZOSERTIB, VE-821, SCH-900776

---

## 🤖 Machine Learning (`machinelearning.py`)

Features : **440 scores d'influence de régulateurs de transcription** (TF) issus de cRegMap  
Validation : **LOO** (Leave-One-Out) pour la classification | **KFold 5** pour la régression

### Partie A — Prédiction du sous-type moléculaire

| Modèle | Accuracy (LOO) |
|---|---|
| Random Forest | **84.5%** |
| Régression logistique | **85.9%** |

✅ Les scores TF capturent bien l'identité moléculaire des sous-types — cohérent avec la biologie cRegMap.

### Partie B — Prédiction de l'AUC par médicament

| Médicament | Random Forest R² | ElasticNet R² | Régression lin. R² |
|---|---|---|---|
| DECITABINE | 0.051 | -0.382 | 0.111 |
| LY2603618 | **0.298** | -0.187 | -0.111 |
| VER-49009 | -0.272 | -0.601 | -1.160 |
| BAY-11-7085 | -1.784 | -1.303 | -5.659 |
| IDAZOXAN | -0.683 | -4.776 | -0.346 |

> ⚠️ Les R² faibles s'expliquent par la dimensionnalité élevée (440 TF) rapportée
> au faible effectif (n=30). Ce phénomène est classique en pharmacogénomique sur
> petites cohortes. LY2603618 (R²=0.298 avec RF) montre le signal le plus prometteur.

---

## ✅ To-do list

### 🔴 Priorité haute
- [x] Filtrage et nettoyage des lignées breast cancer (DepMap)
- [x] Classification par sous-types moléculaires (cRegMap)
- [x] PCA + visualisations
- [x] Top médicaments par sous-type
- [x] Tests statistiques (Kruskal-Wallis + Mann-Whitney)
- [x] ML — classification sous-types (RF + régression logistique)
- [x] ML — prédiction AUC (RF + ElasticNet + régression linéaire)

### 🟠 Extensions
- [ ] Interprétation biologique des 35 médicaments significatifs

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
| influence_uploadedData | [brcaregmap](https://brcaregmap-781093644550.europe-west1.run.app/) | Scores d'influence des régulateurs de transcription (440 TF) |

---

## 👥 Équipe

Projet réalisé en binôme 

AMYAY AMAL — [@melamyay]
COKELAER ALEXIS — [@alexiscokelaer]

