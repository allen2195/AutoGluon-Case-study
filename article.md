# AutoGluon Trained Itself While I Questioned Whether That Was a Good Idea

*A case study on heart disease prediction — and what happens when you let a library make all the decisions.*

---

I picked the Heart Disease UCI dataset partly because it's clean and well-understood, and partly because cardiovascular disease is the kind of problem where bad predictions have real consequences. It felt like the right thing to test an AutoML tool on — not MNIST, not house prices, something that matters. The dataset has 303 patients, 13 routine clinical measurements (blood pressure, cholesterol, chest pain type, resting ECG, a few others), and a binary outcome: disease present or absent.

My actual question wasn't "can AutoGluon do better than me." It was: *when it does, do I understand why, and does that matter?*

---

## What AutoGluon does, briefly

AutoGluon is Amazon's open-source AutoML library for tabular data. The pitch is simple: give it a CSV, tell it the label column, and it handles the rest — feature preprocessing, model selection, hyperparameter tuning, stacking, ensembling. The `best_quality` preset specifically does multi-layer stacking: it trains a portfolio of base models (random forests, extra trees, gradient boosting variants, neural nets), then trains *stacker models* that treat the out-of-fold predictions as new features, then builds a weighted ensemble on top of that. It's a lot of machinery for what is, in the end, a greedy search over a large model space.

I also ran `optimize_for_deployment`, which skips the full stack and produces a leaner model — faster to serve, less accurate.

---

## What it actually produced

One thing I did not expect: AutoGluon on this dataset never trained a LightGBM or XGBoost model. The entire leaderboard was ExtraTrees and RandomForest variants. I'd assumed gradient boosting would dominate the way it usually does on tabular data, but apparently the time limit and the small dataset size pushed it toward tree ensembles that fit faster. The best model was `ExtraTrees_r126_BAG_L1` — a bagged ExtraTreesClassifier, not the fancy stacked neural net I'd pictured.

Results on the held-out test set:

| Approach | ROC-AUC |
|----------|---------|
| AutoGluon `best_quality` | **~0.93** |
| AutoGluon `optimize_for_deployment` | ~0.92 |
| Scikit-learn Random Forest (GridSearchCV) | ~0.91 |
| Scikit-learn Logistic Regression | ~0.88 |

Honestly, the gap is smaller than I expected. AutoGluon wins, but on 303 rows with 13 features, a well-tuned Random Forest is not far behind. I'm a bit skeptical that the stacking layers are actually doing much here — on a dataset this small, the "ensemble of stacked models" is probably mostly capturing variance in the cross-validation folds rather than learning genuinely new structure. A larger dataset would test this properly.

The SHAP analysis pointed to `cp` (chest pain type), `thal` (thalassemia), and `ca` (number of major vessels) as the dominant features. That matches what cardiologists actually look at, which is reassuring — the model isn't winning by memorizing something spurious.

---

## Compared to MLJAR and scikit-learn

MLJAR sits between AutoGluon and a hand-built sklearn pipeline. It gives you a configurable model portfolio, generates explainability reports automatically, and doesn't hide what it's doing. For anyone who needs to show their work — a regulator, a supervisor, a client — MLJAR's transparency is genuinely useful. AutoGluon produces a better number and tells you almost nothing about how it got there.

The sklearn pipeline took me about 30 minutes to set up properly: imputation strategy, encoding choices, writing the GridSearchCV grid. AutoGluon took five minutes and beat it. That's the trade-off in plain terms. What you give up is the understanding. With the sklearn pipeline, I could explain every decision. With AutoGluon, I had to bolt SHAP on afterward and hope the explanation holds.

---

## Would I use it in production? It depends on what "production" means.

For prototyping and benchmarking, AutoGluon is excellent. Run it first, see what the performance ceiling looks like, then decide whether it's worth engineering a proper system. If AutoGluon barely beats a logistic regression, that tells you something important about the data before you spend weeks on it.

For a deployed system, my honest answer is: probably not the `best_quality` ensemble, and not in anything regulated. The ensemble is slow to serve — you're running multiple stacked models per prediction. In healthcare or credit, you also typically need to explain individual predictions to someone, and "a weighted combination of 8 bagged ExtraTrees models" is not an explanation. `optimize_for_deployment` is more realistic, but even then I'd want to understand what's inside it before putting it in front of patients.

The harder question is distribution shift. AutoGluon optimizes hard for its validation metric. On a new hospital's data, or a slightly different patient population, a simpler model might hold up better precisely because it hasn't overfit to the patterns in this particular 303-row slice. I don't know the answer to that — I'd need more data to test it — but it's the thing I'd worry about.

Use AutoGluon to find out what's possible. Don't use it as a reason to stop thinking.

---

*Code and full notebook at: [github.com/YOUR_USERNAME/autogluon-heart-disease](https://github.com/)*
