# Titanic Survival Analysis
### End-to-end data science portfolio — Dhivya Madhavan

A rigorous, analytical workflow applied to the Kaggle Titanic dataset. Every methodological decision is justified by a specific analytical finding — not convention.

---

## What this is

This is not a tutorial. It is a demonstration of the full data science workflow from raw data to comparative model evaluation, with explicit reasoning at every step.

**Central question:** What combination of passenger characteristics best predicts survival — and which factors are causally interpretable vs merely correlated?

---

## Key findings

**The dataset itself had a problem.** The Kaggle test set showed 100%/0% female/male survival — a perfect separator (r = −1.0). Any model trained on this learns one rule. All modeling was conducted on the training set, which reflects real historical outcomes (74.2% / 18.9%).

**3rd class women survived at only 50%.** Logistic regression predicted high survival for all women (female → survive, linearly). Random forest discovered the conditional: female × 1st class = 97%, female × 3rd class = 50%. A 47pp gap within the same sex.

**LogFare was simultaneously insignificant and the 2nd most important feature.** LR: p = 0.604 once Pclass is in the model. RF: importance = 24.6%. Both correct — they measure different things. LR measures marginal linear contribution. RF measures non-linear interactive signal across all splits.

**Copula analysis justified the interaction term.** Gaussian copula ρ = 0.227 among survivors vs 0.041 among non-survivors — 5× difference. Survivors' age and wealth were structurally coupled in a way non-survivors' weren't. Mathematical justification for an Age × Fare interaction, before any model was built.

---

## Model results

| Model | CV Accuracy | CV AUC | Recall | False Negatives |
|---|---|---|---|---|
| Logistic Regression | 79.0% ±1.4% | 0.853 ±0.024 | 70.8% | 100 |
| Random Forest | **84.2% ±1.1%** | **0.879 ±0.019** | 79.5% | 70 |
| XGBoost | 84.0% ±1.6% | 0.878 ±0.020 | **86.5%** | **46** |

Use LR for interpretability (odds ratios). Use RF for stable production prediction. Use XGBoost when recall matters most.

---

## Multivariate methods applied

| Method | What it measured | Decision it produced |
|---|---|---|
| Scatter plots | Visual separation by variable | Fare separates; Age does not |
| KDE | Distribution shape by survival group | Age child-bump → engineer IsChild |
| Correlation matrix | Linear relationships + multicollinearity | Drop SibSp, Parch, FamilySize |
| Covariance matrix | Scale-dependent relationships | Age inflates; use correlation for interpretation |
| Joint distribution | 2D combined density (Age × Fare) | Dual-peak survivor structure |
| Copula analysis | Dependency structure (margin-free) | Survivor dependency 5× non-survivor → interaction term |
| PCA | Orthogonal variance directions | Sex is orthogonal to class/wealth — independent dimension |

---

## Repository structure

```
titanic-survival-analysis/
├── notebooks/
│   └── titanic_analysis_v5.ipynb   # 55 cells, full workflow
├── docs/
│   └── titanic_case_study.docx     # 7-section portfolio case study
├── data/
│   └── (download train.csv from Kaggle — link below)
└── README.md
```

---

## Running the notebook

1. Download `train.csv` from [kaggle.com/c/titanic/data](https://www.kaggle.com/c/titanic/data)
2. Open `notebooks/titanic_analysis_v5.ipynb` in [Google Colab](https://colab.research.google.com)
3. Upload `train.csv` when prompted in cell 2
4. Run all cells in order

**Requirements:** All libraries (pandas, numpy, scipy, sklearn, matplotlib, seaborn) are pre-installed in Colab.

---

## Feature engineering decisions

| Feature | Source | Justification |
|---|---|---|
| `LogFare` | `log(Fare + 1)` | KDE showed right-skewed raw fare; log-transform justified |
| `IsChild` | `Age < 10` | KDE child survival bump; non-linear — raw Age misses it |
| `IsAlone` | `FamilySize == 1` | r=−0.203 with survival; more interpretable than FamilySize |
| `Sex_num` | `male = 1` | r=−0.543; strongest predictor; orthogonal to class (PCA) |
| Title-group age imputation | Name → Title → median | Preserves demographic signal; global median systematically wrong |

**Dropped:** SibSp (r=+0.891 with FamilySize), Parch (r=+0.783 with FamilySize), FamilySize (IsAlone more interpretable)

---

## About

Dhivya Madhavan
[LinkedIn](https://linkedin.com/in/dhivya-madhavan) | [Kaggle](https://kaggle.com)

> *"EDA is not decoration. Every chart should produce a 'therefore' that changes how you build the model."*
