# ML Assignment 2 — Student Dropout & Academic Success Classifier

## a. Problem statement
Higher-education institutions lose students to dropout, and intervening early needs a way to flag at-risk students.
Using information known at enrolment plus first- and second-semester academic performance, predict whether a student
will **Drop out**, stay **Enrolled** (still studying after the normal duration), or **Graduate**: a 3-class
classification problem. Five classifiers are trained and compared on the same data, and a Streamlit app lets anyone
upload test data, pick a saved model and see its metrics, confusion matrix and classification report.

## b. Dataset description
- **Source:** UCI Machine Learning Repository, *Predict Students' Dropout and Academic Success* (ID 697),
  https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
- **Instances:** 4,424 students (Polytechnic Institute of Portalegre, Portugal)
- **Features:** 36 (all numeric / integer-encoded), no missing values
  - Demographic: marital status, nationality, gender, age at enrolment, displaced, international
  - Socio-economic: mother's/father's qualification and occupation, debtor, tuition fees up to date, scholarship holder
  - Academic path: application mode/order, course, previous qualification (and grade), admission grade
  - Performance: curricular units credited / enrolled / evaluated / approved / grade / without evaluation, for the 1st and 2nd semester
  - Macro-economic: unemployment rate, inflation rate, GDP
- **Target:** `Target` ∈ {Dropout (1,421), Enrolled (794), Graduate (2,209)}, so the classes are imbalanced
- **Split:** 80 % train / 20 % test, stratified, `random_state=7` → `test_data.csv` has 885 rows (36 features + Target)

## c. GitHub repository link
https://github.com/PriyanshuKanyal37/ml-assignment2-student-dropout

```
├── app.py               # Streamlit app (upload CSV → choose model → metrics + confusion matrix + report)
├── train_models.py      # trains the 5 models, saves them, writes metrics + test_data.csv
├── test_app.py          # smoke check: app renders and saved models reproduce metrics.csv
├── requirements.txt
├── test_data.csv        # 20 % hold-out test set used for all reported metrics
├── raw/data.csv         # original UCI dataset (semicolon separated)
└── model/               # 5 saved .joblib models + metrics.csv
```

## d. Models used
All five models are trained on the same 80 % split. Logistic Regression, KNN, Naive Bayes and SVM are sklearn
`Pipeline`s with `StandardScaler`, so the scaler is saved inside each model and the app needs no separate
preprocessing file. Multi-class metrics: Precision / Recall / F1 are **weighted** averages, AUC is **one-vs-rest weighted**.

| Model | Settings |
|---|---|
| Logistic Regression | StandardScaler + LogisticRegression(max_iter=2000) |
| Decision Tree | DecisionTreeClassifier(max_depth=8, min_samples_leaf=10) |
| KNN | StandardScaler + KNeighborsClassifier(n_neighbors=15) |
| Naive Bayes | StandardScaler + GaussianNB |
| Support Vector Machine | StandardScaler + SVC(kernel="rbf", C=1, probability=True) |

### Comparison table (hold-out test set, 885 rows)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | **0.7729** | **0.9031** | **0.7591** | **0.7729** | **0.7608** | **0.6230** |
| Decision Tree | 0.7638 | 0.8720 | 0.7501 | 0.7638 | 0.7506 | 0.6075 |
| KNN | 0.7040 | 0.8413 | 0.6934 | 0.7040 | 0.6777 | 0.5058 |
| Naive Bayes | 0.6768 | 0.8037 | 0.6568 | 0.6768 | 0.6619 | 0.4588 |
| Support Vector Machine | 0.7492 | 0.8954 | 0.7365 | 0.7492 | 0.7327 | 0.5824 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best on every metric (Acc 0.773, AUC 0.903, MCC 0.623). The strongest predictors (approved units and grades in semesters 1 and 2) relate to the outcome almost monotonically, so after scaling a linear decision boundary separates Dropout from Graduate well. Its weak spot, shared by all models, is the minority *Enrolled* class, which sits between the other two. |
| Decision Tree | A close second on accuracy (0.764) and MCC (0.608). It learns clear rules such as "few 2nd-semester approved units → Dropout". Limiting depth to 8 and leaves to ≥10 samples stopped overfitting, but its probability estimates are coarse (one value per leaf), so its AUC (0.872) trails LR and SVM. |
| KNN | Mid-to-low (Acc 0.704, MCC 0.506). Many features are integer codes (course, occupation, application mode) whose numeric distance means nothing, so in 36 dimensions the nearest neighbours are noisy. It also leans toward the majority *Graduate* class, so it has the lowest recall on *Enrolled*. |
| Naive Bayes | Weakest (Acc 0.677, MCC 0.459). Its independence assumption is badly broken: semester-1 and semester-2 features are highly correlated, so the same evidence is counted twice. The categorical codes and count features are also far from Gaussian. It is fast and gives a useful AUC baseline (0.804). |
| Support Vector Machine | Second-best AUC (0.895) and third on accuracy (0.749). The RBF kernel ranks the classes well, but with default C and gamma (untuned) it does not beat the simpler linear model here. It is also the slowest model to train and predict, because `probability=True` adds internal cross-validation. |
| **Overall, winner for this dataset** | **Logistic Regression**: highest Accuracy, AUC, F1 and MCC, plus it is fast and interpretable. SVM and Decision Tree are close behind; Naive Bayes is the poorest fit for this correlated, mixed-type data. |

## Streamlit app
- **Live app:** _<paste Streamlit Community Cloud URL here>_
- Features: CSV upload (test data), model selection dropdown, the 6 evaluation metrics, confusion matrix heat-map,
  classification report, row-level predictions, a side-by-side comparison tab and a sample `test_data.csv` download.
- The app only **loads** the saved models from `model/`. It never retrains on the uploaded data.

## Run locally
```bash
pip install -r requirements.txt
python train_models.py      # optional: re-creates model/ and test_data.csv
streamlit run app.py
```
