import pandas as pd
from collections import Counter
from scipy.stats import chi2_contingency

# Load data
df = pd.read_csv("boxes.csv")

# Map outcome codes for clarity
# 1 = undemonstrated option, 2 = majority option, 3 = minority option

n = len(df)
majority_n = (df["y"] == 2).sum()
minority_n = (df["y"] == 3).sum()
undemo_n = (df["y"] == 1).sum()

print("Total N:", n)
print("Majority choices:", majority_n, f"({majority_n / n:.3f})")
print("Minority choices:", minority_n, f"({minority_n / n:.3f})")
print("Undemonstrated choices:", undemo_n, f"({undemo_n / n:.3f})")

# Majority preference by culture
print("\nMajority choice rate by culture:")
print(df.groupby("culture")["y"].apply(lambda s: (s == 2).mean()))

# Majority preference by age (treating age as integer years)
print("\nMajority choice rate by age:")
print(df.groupby("age")["y"].apply(lambda s: (s == 2).mean()))

# Simple contrasts: majority vs minority overall
counts = Counter(df["y"])

majority = counts[2]
minority = counts[3]

print("\nMajority vs minority difference (count):", majority - minority)

# Overall rates
majority_rate = majority_n / n
minority_rate = minority_n / n

print("\nOverall majority rate:", majority_rate)
print("Overall minority rate:", minority_rate)

# Chi-square tests for variation across culture and age
print("\n=== Chi-square tests ===")
ct_culture = pd.crosstab(df["culture"], df["y"])
chi2_cult, p_cult, dof_cult, exp_cult = chi2_contingency(ct_culture)
print("Outcome x culture: chi2 =", chi2_cult, "df =", dof_cult, "p =", p_cult)

ct_age = pd.crosstab(df["age"], df["y"])
chi2_age, p_age, dof_age, exp_age = chi2_contingency(ct_age)
print("Outcome x age:     chi2 =", chi2_age, "df =", dof_age, "p =", p_age)

# Binary social-information use (any demonstrated option vs undemonstrated)
df["social"] = df["y"].isin([2, 3]).astype(int)

ct_social_culture = pd.crosstab(df["culture"], df["social"])
chi2_sc, p_sc, dof_sc, _ = chi2_contingency(ct_social_culture)
print("\nSocial (demonstrated) vs undemonstrated x culture:",
      "chi2 =", chi2_sc, "df =", dof_sc, "p =", p_sc)

ct_social_age = pd.crosstab(df["age"], df["social"])
chi2_sa, p_sa, dof_sa, _ = chi2_contingency(ct_social_age)
print("Social (demonstrated) vs undemonstrated x age:",
      "chi2 =", chi2_sa, "df =", dof_sa, "p =", p_sa)

# Simple variability summaries for majority choice rate
maj_by_culture = df.groupby("culture")["y"].apply(lambda s: (s == 2).mean())
maj_by_age = df.groupby("age")["y"].apply(lambda s: (s == 2).mean())

print("\nStd of majority rate across cultures:", maj_by_culture.std())
print("Min/max majority rate across cultures:", maj_by_culture.min(), maj_by_culture.max())

print("\nStd of majority rate across ages:", maj_by_age.std())
print("Min/max majority rate across ages:", maj_by_age.min(), maj_by_age.max())
