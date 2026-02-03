import pandas as pd
from statsmodels.stats.weightstats import ttest_ind

# Load data
path = "affairs.csv"
df = pd.read_csv(path)

# The column names are shuffled; infer true variables by value patterns.
# 'religiousness' is binary yes/no -> children indicator
# 'age' has values {0,1,2,3,7,12} -> affairs frequency
children = df["religiousness"]
affairs = df["age"]

# Group summaries
summary = (
    df.assign(children=children, affairs=affairs)
    .groupby("children")
    .agg(
        n=("affairs", "size"),
        mean_affairs=("affairs", "mean"),
        median_affairs=("affairs", "median"),
        any_affair_rate=("affairs", lambda x: (x > 0).mean()),
    )
)

# T-test for difference in mean affairs (Welch)
with_children = affairs[children == "yes"]
without_children = affairs[children == "no"]

# statsmodels returns (tstat, pvalue, df)
tstat, pval, _ = ttest_ind(with_children, without_children, usevar="unequal")

print("Summary by children (yes/no):")
print(summary)
print("\nWelch t-test for mean affairs (yes vs no):")
print(f"t = {tstat:.3f}, p = {pval:.4f}")

# Difference in means
mean_diff = with_children.mean() - without_children.mean()
print(f"\nMean difference (yes - no): {mean_diff:.3f}")
