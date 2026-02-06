import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data
path = "affairs.csv"
df = pd.read_csv(path)

# Identify key variables
# feature2: frequency of extramarital intercourse in past year (count-like)
# feature6: children in marriage (yes/no)
# feature3: gender (female/male)

affairs = df["feature2"].astype(float)
children = (df["feature6"].astype(str).str.lower() == "yes").astype(int)

# Basic grouping stats
stats = (
    df.assign(has_children=children, affairs=affairs)
      .groupby("has_children")
      .agg(
          n=("affairs", "size"),
          mean_affairs=("affairs", "mean"),
          median_affairs=("affairs", "median"),
          prop_any_affair=("affairs", lambda x: (x > 0).mean()),
      )
)

# Differences: children yes (1) minus no (0)
mean_diff = stats.loc[1, "mean_affairs"] - stats.loc[0, "mean_affairs"]
prop_diff = stats.loc[1, "prop_any_affair"] - stats.loc[0, "prop_any_affair"]

# Bootstrap CIs for differences
rng = np.random.default_rng(0)
B = 1000
mean_diffs = np.empty(B)
prop_diffs = np.empty(B)

for b in range(B):
    idx = rng.integers(0, len(df), len(df))
    samp = df.iloc[idx].copy()
    s_affairs = samp["feature2"].astype(float)
    s_children = (samp["feature6"].astype(str).str.lower() == "yes").astype(int)
    g = (
        samp.assign(has_children=s_children, affairs=s_affairs)
           .groupby("has_children")
           .agg(
               mean_affairs=("affairs", "mean"),
               prop_any_affair=("affairs", lambda x: (x > 0).mean()),
           )
    )
    # Ensure both groups exist in resample
    if 0 in g.index and 1 in g.index:
        mean_diffs[b] = g.loc[1, "mean_affairs"] - g.loc[0, "mean_affairs"]
        prop_diffs[b] = g.loc[1, "prop_any_affair"] - g.loc[0, "prop_any_affair"]
    else:
        mean_diffs[b] = np.nan
        prop_diffs[b] = np.nan

mean_ci = np.nanpercentile(mean_diffs, [2.5, 97.5])
prop_ci = np.nanpercentile(prop_diffs, [2.5, 97.5])

# Logistic regression for any affair > 0
# Controls: gender, age, years married, religiousness, education, occupation, marriage rating
model_df = df.copy()
model_df["has_children"] = children
model_df["male"] = (model_df["feature3"].astype(str).str.lower() == "male").astype(int)
model_df["any_affair"] = (model_df["feature2"].astype(float) > 0).astype(int)

X = model_df[["has_children", "male", "feature4", "feature5", "feature7", "feature8", "feature9", "feature10"]].astype(float)
X = sm.add_constant(X)
y = model_df["any_affair"].astype(int)

logit = sm.Logit(y, X)
res = logit.fit(disp=False)

coef = res.params["has_children"]
se = res.bse["has_children"]
pval = res.pvalues["has_children"]
ci_low, ci_high = res.conf_int().loc["has_children"].tolist()

odds_ratio = float(np.exp(coef))
or_ci_low, or_ci_high = float(np.exp(ci_low)), float(np.exp(ci_high))

# Report
print("Group stats (has_children=1 means children present):")
print(stats)
print("\nDifferences (children - no children):")
print(f"Mean affairs diff: {mean_diff:.4f} (95% CI {mean_ci[0]:.4f}, {mean_ci[1]:.4f})")
print(f"Prop any affair diff: {prop_diff:.4f} (95% CI {prop_ci[0]:.4f}, {prop_ci[1]:.4f})")
print("\nLogistic regression on any affair (controls included):")
print(f"has_children coef: {coef:.4f}, SE {se:.4f}, p={pval:.4g}")
print(f"Odds ratio: {odds_ratio:.4f} (95% CI {or_ci_low:.4f}, {or_ci_high:.4f})")
