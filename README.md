# Task 2: Exploratory Data Analysis (EDA) – Titanic Dataset

**Elevate Labs – AI & ML Internship**

## Objective
Understand the data using descriptive statistics and visualizations.

## Tools
Python, Pandas, NumPy, Matplotlib, Seaborn, Plotly

## Files
| File | Description |
|---|---|
| `eda.py` | Complete EDA script (run: `python eda.py`) |
| `Titanic-Dataset.csv` | Raw dataset (891 rows × 12 columns) |
| `summary_statistics.csv` | Mean / median / std / skew / kurtosis of numeric features |
| `output_log.txt` | Console output of the run |
| `screenshots/` | 6 static plots (histograms, boxplots, pairplot, correlation matrix, survival patterns) |
| `interactive/` | 4 interactive Plotly charts (open the `.html` files in a browser) |

## What I Did
1. **Summary statistics** – mean, median, std, quartiles, skewness, kurtosis, missing values.
2. **Histograms & boxplots** – for `Age`, `Fare`, `SibSp`, `Parch`, plus boxplots grouped by `Survived` and `Pclass`.
3. **Pairplot & correlation matrix** – relationships between features and with `Survived`.
4. **Patterns, trends & anomalies** – survival by sex, class, port, age group, family size.
5. **Feature-level inferences** – listed below.
6. **Plotly** – interactive versions of key charts.

## Key Findings

### Summary statistics
| Feature | Mean | Median | Std | Skew | Comment |
|---|---|---|---|---|---|
| Age | 29.70 | 28.0 | 14.53 | 0.39 | Roughly symmetric; 177 missing (19.9%) |
| Fare | 32.20 | 14.45 | 49.69 | 4.79 | Heavily right-skewed (mean >> median) |
| SibSp | 0.52 | 0 | 1.10 | 3.70 | Most travelled without siblings/spouse |
| Parch | 0.38 | 0 | 0.81 | 2.75 | Most travelled without parents/children |

Overall survival rate: **38.4%**.

### Patterns & inferences
- **Sex is the strongest predictor.** Female survival 74.2% vs male 18.9% (corr with Survived = +0.54).
- **Class matters.** Survival: 1st 63.0%, 2nd 47.3%, 3rd 24.2% (corr = −0.34).
- **Sex × Class interaction.** 1st-class women survived 96.8%, 3rd-class men only 13.5%. Sex and class together explain most of the pattern.
- **Fare is positively related to survival** (corr +0.26), but largely because Fare tracks class (Fare–Pclass corr = −0.55).
- **Age:** children (≤12) had 58% survival; seniors (60+) only 22.7%. Age alone is only weakly correlated (−0.08), so the effect is non-linear and concentrated among children.
- **Port of embarkation:** Cherbourg 55.4% > Queenstown 39.0% > Southampton 33.7%. Likely a proxy for class mix (Cherbourg had more 1st-class passengers), not a direct cause.
- **Family size:** travelling alone had 30.4% survival; families of 2–4 had 55–72%; families of 5+ dropped sharply (0–33%).

### Anomalies / data quality
- `Fare` is extremely skewed: 116 IQR outliers, max £512.33 (3 tickets > £500).
- 15 passengers have `Fare = 0` (possibly crew/staff or missing data).
- `Cabin` is 77.1% missing; `Age` 19.9%; `Embarked` 2 rows.
- Missing `Age` is not random: those passengers survived less (29.4% vs 40.6%) and are mostly 3rd class.
- `Age` median differs by class (1st: 37, 2nd: 29, 3rd: 24) → class-wise age imputation would be better than one global median.
- `Parch` shows 213 IQR "outliers", but they are legitimate values (passengers with parents/children), not errors.

### Multicollinearity check
Strongest feature–feature correlations: `Pclass`–`Fare` (−0.55), `SibSp`–`Parch` (0.41), `Pclass`–`Age` (−0.37). None exceed 0.8, so there is no severe multicollinearity, but `Pclass`/`Fare` carry overlapping information.

### Suggested next steps for modelling
Log-transform `Fare`, create `FamilySize` / `IsAlone` / `Title` features, impute `Age` by class (or title), and encode `Sex` and `Embarked`.

---

## Interview Questions & Answers

**1. What is the purpose of EDA?**
To understand the data before modelling: its structure, distributions, missing values, outliers, relationships between variables, and data quality issues. It helps form hypotheses, choose preprocessing steps and features, and pick suitable models.

**2. How do boxplots help in understanding a dataset?**
They show the median, quartiles (Q1/Q3), IQR, whiskers, and outliers in one compact picture. They reveal spread, skewness (median position within the box) and make it easy to compare distributions across groups (e.g., Fare by Pclass). Weakness: they hide multimodality, which histograms/violin plots show.

**3. What is correlation and why is it useful?**
Correlation measures the strength and direction of a (usually linear) relationship between two variables, from −1 to +1 (Pearson). It is useful for finding predictive features, spotting redundant features, and detecting multicollinearity. Correlation does not imply causation, and Pearson misses non-linear relationships (use Spearman or plots).

**4. How do you detect skewness in data?**
Histogram/KDE plots (long tail on one side), mean vs median comparison, boxplots (asymmetric whiskers), and the numeric `df.skew()` value. Skew near 0 is symmetric; |skew| > 1 is highly skewed. In this dataset `Fare` has skew 4.79 (right-skewed). Fix with log, square-root or Box-Cox transforms.

**5. What is multicollinearity?**
When two or more independent features are highly correlated with each other, so they carry redundant information. It makes coefficients in linear/logistic regression unstable and hard to interpret. Detect with the correlation matrix or Variance Inflation Factor (VIF > 5–10 is a concern). Fix by dropping/combining features, PCA, or regularization (Ridge/Lasso).

**6. What tools do you use for EDA?**
Pandas (`describe`, `info`, `groupby`, `corr`), NumPy, Matplotlib, Seaborn, Plotly for interactive charts, Jupyter Notebook, and automated profiling libraries such as ydata-profiling (pandas-profiling) and Sweetviz. Excel/Power BI/Tableau are also used.

**7. Can you explain a time when EDA helped you find a problem?**
In this Titanic EDA, plotting missing values and survival by missingness showed that `Age` was not missing at random: passengers with missing age survived less (29% vs 41%) and were mostly 3rd class. Filling every gap with one global mean would have hidden that pattern, so I'd impute by class instead. EDA also revealed the extreme `Fare` skew and 15 zero-fare records that needed handling.
*(Tip: in a real interview, use your own project example if you have one.)*

**8. What is the role of visualization in ML?**
It helps understand data (distributions, outliers, class imbalance), choose features, diagnose problems (e.g., learning curves for over/underfitting, confusion matrices, ROC curves), interpret models (feature importance, SHAP) and communicate results to non-technical stakeholders.

---
*Submitted as part of Elevate Labs AI & ML Internship – Task 2.*
