# AutoGluon vs. The Rest: Can a Framework That Trains Itself Beat Careful Manual Work?

A case study using the **Heart Disease UCI dataset** predicting cardiovascular risk with AutoGluon's AutoML pipeline, then interrogating it with SHAP and comparing it against a hand-tuned scikit-learn baseline.

---

## Project Structure

```
autogluon-project/
├── notebooks/
│   └── autogluon_heart_disease.ipynb   # Full pipeline: EDA → AutoGluon → SHAP → comparison
├── visuals/
│   ├── leaderboard.png                 # AutoGluon model leaderboard bar chart
│   ├── feature_importance_shap.png     # SHAP beeswarm plot
│   └── confusion_matrix.png           # Best model confusion matrix
├── article.md                          # Medium-style technical write-up
├── requirements.txt                    # Python dependencies
└── README.md
```

---

## The Problem

Heart disease remains the leading cause of death globally. Early, accurate prediction of cardiovascular risk from routine clinical measurements  blood pressure, cholesterol, chest pain type can meaningfully change patient outcomes. This project asks: **how quickly and how well can AutoGluon solve this problem, right out of the box?**

**Dataset:** [Heart Disease UCI](https://www.openml.org/d/53) via OpenML  
**Task:** Binary classification — predict presence or absence of heart disease  
**Target metric:** ROC-AUC (chosen because class balance is imperfect and clinical cost of false negatives is high)

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/allen2195/AutoGluon-Case-study.git
cd autogluon-heart-disease
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** AutoGluon has large optional dependencies. On Apple Silicon, install via conda for best compatibility. On Windows, WSL2 is recommended.

### 4. Launch the notebook

```bash
jupyter notebook notebooks/autogluon_heart_disease.ipynb
```

---

## Key Findings

| Model | ROC-AUC (val) | Notes |
|-------|---------------|-------|
| AutoGluon `best_quality` ensemble | **0.934** | Weighted ensemble of 8+ models |
| AutoGluon `optimize_for_deployment` | 0.921 | 4× faster inference |
| Scikit-learn (tuned Random Forest) | 0.906 | GridSearchCV, 30 min tuning |
| Scikit-learn (logistic regression baseline) | 0.876 | 5-min setup |

AutoGluon's stacked ensemble outperformed a carefully tuned Random Forest by ~2.8 AUC points, in roughly the same wall-clock time once you account for manual grid search.

**Top predictive features (SHAP):** `cp` (chest pain type), `thal` (thalassemia type), `ca` (number of major vessels), `oldpeak` (ST depression)

---

## Reproducibility

All random seeds are fixed (`random_state=42`). The notebook is designed to run top-to-bottom without manual intervention. AutoGluon downloads no external assets at fit time — only at install time.

Expected full runtime: **~8–12 minutes** on a standard laptop (CPU only).

---

## Article

The companion write-up in [`article.md`](./article.md) covers:
- What AutoGluon actually does under the hood
- Why this dataset, and what the numbers mean clinically
- A structured comparison with scikit-learn pipelines and MLJAR
- An honest production assessment — when to use it, when not to

---

## License

MIT. Dataset is public domain (UCI / OpenML).
