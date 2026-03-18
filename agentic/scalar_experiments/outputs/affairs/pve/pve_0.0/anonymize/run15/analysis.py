import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

DATA_PATH = "affairs.csv"

df = pd.read_csv(DATA_PATH)

# Map columns
# feature2: affairs frequency
# feature6: children (yes/no)

# Basic cleaning
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].str.strip()

# Encode children yes=1 no=0
children = df["feature6"].map({"yes": 1, "no": 0})

# Outcome: affairs frequency
affairs = df["feature2"].astype(float)

# Any affair indicator
any_affair = (affairs > 0).astype(int)

# Group stats
stats_by_child = df.assign(children=children, affairs=affairs).groupby("children")["affairs"].agg(["count", "mean", "median", "std"])\
    .rename(index={0: "no_children", 1: "children"})

# t-test for mean difference
affairs_child = affairs[children == 1]
affairs_nochild = affairs[children == 0]

ttest = stats.ttest_ind(affairs_child, affairs_nochild, equal_var=False, nan_policy="omit")

# Mann-Whitney U
mw = stats.mannwhitneyu(affairs_child, affairs_nochild, alternative="less")

# Logistic regression for any affair ~ children + controls
# Controls: gender, age, years married, religiousness, education, occupation, marriage rating
controls = pd.DataFrame({
    "children": children,
    "female": (df["feature3"].str.lower() == "female").astype(int),
    "age": df["feature4"].astype(float),
    "years_married": df["feature5"].astype(float),
    "religious": df["feature7"].astype(float),
    "education": df["feature8"].astype(float),
    "occupation": df["feature9"].astype(float),
    "marriage_rating": df["feature10"].astype(float),
})

controls = sm.add_constant(controls)

logit_model = sm.Logit(any_affair, controls, missing="drop")
logit_res = logit_model.fit(disp=False)

# OLS for affairs frequency (continuous proxy) ~ children + controls
ols_model = sm.OLS(affairs, controls, missing="drop")
ols_res = ols_model.fit()

summary = {
    "group_stats": stats_by_child.to_dict(),
    "ttest": {
        "statistic": float(ttest.statistic),
        "pvalue": float(ttest.pvalue),
        "mean_children": float(affairs_child.mean()),
        "mean_no_children": float(affairs_nochild.mean()),
    },
    "mannwhitney": {
        "statistic": float(mw.statistic),
        "pvalue": float(mw.pvalue),
    },
    "logit_children": {
        "coef": float(logit_res.params["children"]),
        "pvalue": float(logit_res.pvalues["children"]),
        "odds_ratio": float(np.exp(logit_res.params["children"])),
    },
    "ols_children": {
        "coef": float(ols_res.params["children"]),
        "pvalue": float(ols_res.pvalues["children"]),
    },
}

print(json.dumps(summary, indent=2))
