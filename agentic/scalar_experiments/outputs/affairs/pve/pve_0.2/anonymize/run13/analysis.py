import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main():
    df = pd.read_csv("affairs.csv")

    # Map columns for clarity
    df = df.rename(columns={
        "feature2": "affairs_freq",
        "feature3": "gender",
        "feature4": "age",
        "feature5": "years_married",
        "feature6": "children",
        "feature7": "religiousness",
        "feature8": "education",
        "feature9": "occupation",
        "feature10": "marriage_rating",
    })

    # Binary indicator for children
    df["has_children"] = (df["children"].str.lower() == "yes").astype(int)

    # Outcome variants
    df["any_affair"] = (df["affairs_freq"] > 0).astype(int)
    df["log1p_affairs"] = np.log1p(df["affairs_freq"])

    # Group stats
    group_stats = df.groupby("has_children")["affairs_freq"].agg(["mean", "median", "std", "count"])
    mean_no = group_stats.loc[0, "mean"]
    mean_yes = group_stats.loc[1, "mean"]
    median_no = group_stats.loc[0, "median"]
    median_yes = group_stats.loc[1, "median"]

    # Welch t-test for mean difference
    no_group = df.loc[df["has_children"] == 0, "affairs_freq"]
    yes_group = df.loc[df["has_children"] == 1, "affairs_freq"]
    t_stat, t_p = stats.ttest_ind(no_group, yes_group, equal_var=False, nan_policy="omit")

    # Mann-Whitney U (nonparametric)
    u_stat, u_p = stats.mannwhitneyu(no_group, yes_group, alternative="two-sided")

    # Cohen's d (pooled SD)
    n0, n1 = len(no_group), len(yes_group)
    s0, s1 = no_group.std(ddof=1), yes_group.std(ddof=1)
    pooled_sd = np.sqrt(((n0 - 1) * s0**2 + (n1 - 1) * s1**2) / (n0 + n1 - 2))
    cohens_d = (mean_yes - mean_no) / pooled_sd if pooled_sd > 0 else np.nan

    # OLS on log1p(affairs) with controls
    ols_model = smf.ols(
        "log1p_affairs ~ has_children + age + years_married + religiousness + education + occupation + marriage_rating + C(gender)",
        data=df,
    ).fit(cov_type="HC3")

    # Logistic regression on any affair with controls
    logit_model = smf.logit(
        "any_affair ~ has_children + age + years_married + religiousness + education + occupation + marriage_rating + C(gender)",
        data=df,
    ).fit(disp=False)

    # Extract key results
    ols_coef = ols_model.params.get("has_children", np.nan)
    ols_p = ols_model.pvalues.get("has_children", np.nan)

    logit_coef = logit_model.params.get("has_children", np.nan)
    logit_p = logit_model.pvalues.get("has_children", np.nan)
    logit_or = np.exp(logit_coef) if np.isfinite(logit_coef) else np.nan

    # Summarize direction
    direction = "lower" if mean_yes < mean_no else "higher"

    # Compose response (Likert 0-100)
    # Higher values indicate stronger evidence of a decrease; lower values indicate evidence against a decrease.
    if mean_yes < mean_no:
        if (t_p < 0.05) and (u_p < 0.05) and (ols_p < 0.05) and (logit_p < 0.05):
            response = 80
        elif ((t_p < 0.05) or (u_p < 0.05)) and ((ols_p < 0.05) or (logit_p < 0.05)):
            response = 70
        elif (t_p < 0.1) or (u_p < 0.1) or (ols_p < 0.1) or (logit_p < 0.1):
            response = 60
        else:
            response = 45
    else:
        if (t_p < 0.05) and (u_p < 0.05) and (ols_p < 0.05) and (logit_p < 0.05):
            response = 10
        elif (t_p < 0.05) or (u_p < 0.05) or (ols_p < 0.05) or (logit_p < 0.05):
            response = 20
        else:
            response = 40

    explanation = (
        "Compared people without children to those with children, the mean affair frequency was "
        f"{mean_no:.3f} (no children) vs {mean_yes:.3f} (children), with medians {median_no:.3f} vs {median_yes:.3f}; "
        f"direction is {direction}. Welch t-test p={t_p:.4f}; Mann-Whitney p={u_p:.4f}; Cohen's d={cohens_d:.3f}. "
        "In adjusted models controlling for age, years married, religiousness, education, occupation, marriage rating, and gender, "
        f"the children coefficient was {ols_coef:.3f} in OLS on log1p(affairs) (p={ols_p:.4f}), and the odds ratio for any affair was "
        f"{logit_or:.3f} (logit p={logit_p:.4f})."
    )

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump({"response": int(response), "explanation": explanation}, f)


if __name__ == "__main__":
    main()
