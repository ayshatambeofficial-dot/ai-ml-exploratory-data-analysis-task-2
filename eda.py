"""
Task 2: Exploratory Data Analysis (EDA) - Titanic Dataset
Elevate Labs - AI & ML Internship

Steps:
 1. Summary statistics (mean, median, std, skew, ...)
 2. Histograms and boxplots for numeric features
 3. Pairplot + correlation matrix
 4. Patterns, trends and anomalies
 5. Feature-level inferences
Plots: Matplotlib/Seaborn (PNG) + Plotly (interactive HTML)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

sns.set_theme(style="whitegrid")
os.makedirs("screenshots", exist_ok=True)
os.makedirs("interactive", exist_ok=True)

df = pd.read_csv("Titanic-Dataset.csv")
num_cols = ["Age", "Fare", "SibSp", "Parch"]

# ------------------------------------------------ 1. SUMMARY STATISTICS
print("=" * 60, "\n1. SUMMARY STATISTICS\n", "=" * 60)
print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
summary = df[num_cols].describe().T
summary["median"] = df[num_cols].median()
summary["skew"] = df[num_cols].skew()
summary["kurtosis"] = df[num_cols].kurt()
print("\nNumeric summary:\n", summary.round(2).to_string())
print("\nCategorical summary:\n", df[["Sex", "Embarked", "Pclass", "Survived"]]
      .describe(include="all").to_string())
summary.round(3).to_csv("summary_statistics.csv")

print("\nOverall survival rate: %.1f%%" % (df.Survived.mean() * 100))
for col in ["Sex", "Pclass", "Embarked"]:
    print(f"\nSurvival rate by {col}:\n",
          (df.groupby(col)["Survived"].agg(["mean", "count"])
           .rename(columns={"mean": "survival_rate"}).round(3)))

# --------------------------------------- 2. HISTOGRAMS & BOXPLOTS
print("\n", "=" * 60, "\n2. HISTOGRAMS & BOXPLOTS\n", "=" * 60)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, col in zip(axes.ravel(), num_cols):
    sns.histplot(df[col].dropna(), kde=True, bins=30, ax=ax, color="steelblue")
    ax.axvline(df[col].mean(), color="red", ls="--", label=f"mean={df[col].mean():.1f}")
    ax.axvline(df[col].median(), color="green", ls="-.", label=f"median={df[col].median():.1f}")
    ax.set_title(f"{col} (skew={df[col].skew():.2f})")
    ax.legend()
plt.suptitle("Histograms of Numeric Features")
plt.tight_layout()
plt.savefig("screenshots/01_histograms.png", dpi=120)
plt.close()

fig, axes = plt.subplots(1, 4, figsize=(15, 4))
for ax, col in zip(axes, num_cols):
    sns.boxplot(y=df[col], ax=ax, color="skyblue")
    ax.set_title(col)
plt.suptitle("Boxplots of Numeric Features")
plt.tight_layout()
plt.savefig("screenshots/02_boxplots.png", dpi=120)
plt.close()

# Boxplots by survival / class
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
sns.boxplot(data=df, x="Survived", y="Age", ax=axes[0])
axes[0].set_title("Age vs Survived")
sns.boxplot(data=df, x="Pclass", y="Fare", ax=axes[1])
axes[1].set_title("Fare vs Pclass")
axes[1].set_yscale("log")
sns.boxplot(data=df, x="Pclass", y="Age", ax=axes[2])
axes[2].set_title("Age vs Pclass")
plt.tight_layout()
plt.savefig("screenshots/03_boxplots_by_group.png", dpi=120)
plt.close()

# Outlier counts (IQR)
print("IQR outlier counts:")
for col in num_cols:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    n = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
    print(f"  {col}: {n}")

# ----------------------------- 3. PAIRPLOT & CORRELATION MATRIX
print("\n", "=" * 60, "\n3. PAIRPLOT & CORRELATION\n", "=" * 60)
pp = sns.pairplot(df[["Survived", "Pclass", "Age", "Fare", "SibSp", "Parch"]].dropna(),
                  hue="Survived", palette="Set1", corner=True, plot_kws={"alpha": 0.6, "s": 18})
pp.fig.suptitle("Pairplot (hue = Survived)", y=1.01)
pp.savefig("screenshots/04_pairplot.png", dpi=100)
plt.close("all")

df_corr = df.copy()
df_corr["Sex_code"] = df_corr["Sex"].map({"male": 0, "female": 1})
corr_cols = ["Survived", "Pclass", "Sex_code", "Age", "SibSp", "Parch", "Fare"]
corr = df_corr[corr_cols].corr()
print(corr.round(2).to_string())
print("\nCorrelation with Survived:\n",
      corr["Survived"].drop("Survived").sort_values(key=abs, ascending=False).round(3))
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("screenshots/05_correlation_matrix.png", dpi=120)
plt.close()

# ------------------------ 4. PATTERNS, TRENDS & ANOMALIES
print("\n", "=" * 60, "\n4. PATTERNS, TRENDS, ANOMALIES\n", "=" * 60)
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
sns.countplot(data=df, x="Sex", hue="Survived", ax=axes[0, 0]); axes[0, 0].set_title("Survival by Sex")
sns.countplot(data=df, x="Pclass", hue="Survived", ax=axes[0, 1]); axes[0, 1].set_title("Survival by Class")
sns.countplot(data=df, x="Embarked", hue="Survived", ax=axes[0, 2]); axes[0, 2].set_title("Survival by Embarkation")
sns.histplot(data=df, x="Age", hue="Survived", multiple="stack", bins=25, ax=axes[1, 0])
axes[1, 0].set_title("Age Distribution by Survival")
sns.barplot(data=df, x="Pclass", y="Survived", hue="Sex", ax=axes[1, 1])
axes[1, 1].set_title("Survival Rate: Class x Sex")
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
sns.barplot(data=df, x="FamilySize", y="Survived", ax=axes[1, 2], color="teal")
axes[1, 2].set_title("Survival Rate by Family Size")
plt.tight_layout()
plt.savefig("screenshots/06_survival_patterns.png", dpi=120)
plt.close()

print("Survival rate Class x Sex:\n",
      df.pivot_table(values="Survived", index="Pclass", columns="Sex", aggfunc="mean").round(3))
df["AgeGroup"] = pd.cut(df["Age"], [0, 12, 18, 35, 60, 100],
                        labels=["Child", "Teen", "YoungAdult", "Adult", "Senior"])
print("\nSurvival by age group:\n",
      df.groupby("AgeGroup", observed=True)["Survived"].agg(["mean", "count"]).round(3))
print("\nSurvival by family size:\n",
      df.groupby("FamilySize")["Survived"].agg(["mean", "count"]).round(3))

print("\nAnomalies:")
print("  Fare == 0 rows:", (df.Fare == 0).sum())
print("  Max fare:", df.Fare.max(), "| rows with Fare > 500:", (df.Fare > 500).sum())
print("  Age missing: %.1f%% | Cabin missing: %.1f%%"
      % (df.Age.isnull().mean() * 100, df.Cabin.isnull().mean() * 100))
print("  Infants (Age<1):", (df.Age < 1).sum(), "| Oldest:", df.Age.max())
print("  Median age of missing-Age group by Pclass:\n",
      df.groupby("Pclass")["Age"].median())
print("  Survival: Age missing vs present:\n",
      df.groupby(df.Age.isnull())["Survived"].mean().round(3))

# --------------------------------------------- PLOTLY (interactive)
fig = px.histogram(df, x="Age", color="Survived", barmode="overlay", nbins=30,
                   title="Age Distribution by Survival (interactive)")
fig.write_html("interactive/age_by_survival.html")
fig = px.scatter(df, x="Age", y="Fare", color=df["Survived"].astype(str),
                 symbol="Pclass", hover_data=["Sex", "Pclass"], log_y=True,
                 title="Age vs Fare (log scale), coloured by Survived")
fig.write_html("interactive/age_vs_fare.html")
fig = px.box(df, x="Pclass", y="Fare", color="Sex", log_y=True,
             title="Fare by Class and Sex (log scale)")
fig.write_html("interactive/fare_by_class.html")
fig = px.imshow(corr.round(2), text_auto=True, color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1, title="Correlation Matrix (interactive)")
fig.write_html("interactive/correlation_matrix.html")
print("\nSaved interactive Plotly charts to interactive/")
