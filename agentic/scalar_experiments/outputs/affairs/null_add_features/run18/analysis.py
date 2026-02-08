import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm


def cohen_d(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    nx = len(x)
    ny = len(y)
    if nx < 2 or ny < 2:
        return np.nan
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    if pooled == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / np.sqrt(pooled)


def main():
    df = pd.read_csv("affairs.csv")
    df = df.copy()

    # Ensure children is categorical and normalize values
    df["children"] = df["children"].astype(str).str.strip().str.lower()
    df = df[df["children"].isin(["yes", "no"])].copy()

    # Outcome variables
    df["affair_any"] = (df["affairs"] > 0).astype(int)

    # Group stats
    group = df.groupby("children")
    mean_affairs = group["affairs"].mean()
    prop_any = group["affair_any"].mean()
    n_group = group.size()

    # Difference: yes - no (negative implies lower with children)
    mean_diff = mean_affairs.get("yes", np.nan) - mean_affairs.get("no", np.nan)
    prop_diff = prop_any.get("yes", np.nan) - prop_any.get("no", np.nan)

    # Tests
    yes_affairs = df.loc[df["children"] == "yes", "affairs"]
    no_affairs = df.loc[df["children"] == "no", "affairs"]

    t_res = stats.ttest_ind(yes_affairs, no_affairs, equal_var=False)
    mw_res = stats.mannwhitneyu(yes_affairs, no_affairs, alternative="two-sided")

    # Logistic regression for affair_any with controls
    # Use numeric encodings for categorical variables
    df_model = df.copy()
    df_model["gender_male"] = (df_model["gender"].str.lower() == "male").astype(int)
    df_model["children_yes"] = (df_model["children"] == "yes").astype(int)

    controls = [
        "children_yes",
        "age",
        "yearsmarried",
        "gender_male",
        "religiousness",
        "education",
        "occupation",
        "rating",
    ]

    X = df_model[controls].astype(float)
    X = sm.add_constant(X, has_constant="add")
    y = df_model["affair_any"].astype(int)

    logit = sm.Logit(y, X).fit(disp=False)
    coef_child = logit.params["children_yes"]
    p_child = logit.pvalues["children_yes"]
    or_child = float(np.exp(coef_child))

    # Cohen's d (yes - no)
    d = cohen_d(yes_affairs, no_affairs)

    # Build a scalar score from evidence
    # Base direction: negative diff -> evidence for decrease (Yes)
    score = 0.0

    # Mean affairs difference contribution
    if not np.isnan(mean_diff):
        score += -mean_diff * 10.0  # scale roughly by count differences

    # Proportion difference contribution
    if not np.isnan(prop_diff):
        score += -prop_diff * 100.0  # scale proportion difference

    # Effect size contribution
    if not np.isnan(d):
        score += -d * 20.0

    # Logistic regression evidence
    # If OR < 1 and significant, add strong support
    if or_child < 1:
        score += (1 - or_child) * 40.0
    else:
        score -= (or_child - 1) * 40.0

    if p_child < 0.01:
        score += 15.0
    elif p_child < 0.05:
        score += 8.0
    elif p_child < 0.10:
        score += 3.0

    # t-test significance
    if t_res.pvalue < 0.01:
        score += 10.0
    elif t_res.pvalue < 0.05:
        score += 5.0
    elif t_res.pvalue < 0.10:
        score += 2.0

    # Cap and round
    score = max(-100.0, min(100.0, score))
    score_int = int(np.round(score))

    # Save scalar conclusion only
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(score_int))

    # Print a short report for verification
    print("n by children:", n_group.to_dict())
    print("mean affairs:", mean_affairs.to_dict())
    print("prop any:", prop_any.to_dict())
    print("mean diff (yes-no):", mean_diff)
    print("prop diff (yes-no):", prop_diff)
    print("t-test p:", t_res.pvalue)
    print("mannwhitney p:", mw_res.pvalue)
    print("logit coef child:", coef_child, "p:", p_child, "OR:", or_child)
    print("cohen d:", d)
    print("score:", score_int)


if __name__ == "__main__":
    main()
