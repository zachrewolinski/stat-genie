import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main():
    df = pd.read_csv("affairs.csv")

    # Normalize children column
    df = df.copy()
    df["children_yes"] = (df["children"].str.lower() == "yes").astype(int)

    # Primary outcomes
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    summary = {}
    summary["n_total"] = len(df)
    summary["n_children_yes"] = int(df["children_yes"].sum())
    summary["n_children_no"] = int((1 - df["children_yes"]).sum())

    mean_affairs_yes = df.loc[df["children_yes"] == 1, "affairs"].mean()
    mean_affairs_no = df.loc[df["children_yes"] == 0, "affairs"].mean()
    summary["mean_affairs_yes"] = float(mean_affairs_yes)
    summary["mean_affairs_no"] = float(mean_affairs_no)
    summary["mean_affairs_diff_yes_minus_no"] = float(mean_affairs_yes - mean_affairs_no)

    any_yes = df.loc[df["children_yes"] == 1, "any_affair"].mean()
    any_no = df.loc[df["children_yes"] == 0, "any_affair"].mean()
    summary["any_affair_yes"] = float(any_yes)
    summary["any_affair_no"] = float(any_no)
    summary["any_affair_diff_yes_minus_no"] = float(any_yes - any_no)

    # OLS on affair frequency
    # Controls: gender, age, yearsmarried, religiousness, education, occupation, rating
    ols = smf.ols(
        "affairs ~ children_yes + gender + age + yearsmarried + religiousness + education + occupation + rating",
        data=df,
    ).fit()
    summary["ols_children_coef"] = float(ols.params["children_yes"])
    summary["ols_children_p"] = float(ols.pvalues["children_yes"])

    # Logistic regression on any affair
    logit = smf.logit(
        "any_affair ~ children_yes + gender + age + yearsmarried + religiousness + education + occupation + rating",
        data=df,
    ).fit(disp=False)
    summary["logit_children_coef"] = float(logit.params["children_yes"])
    summary["logit_children_p"] = float(logit.pvalues["children_yes"])

    # Marginal effect for children_yes on probability
    margeff = logit.get_margeff(at="overall")
    me = margeff.summary_frame()
    if "children_yes" in me.index:
        summary["logit_children_margeff"] = float(me.loc["children_yes", "dy/dx"])
    else:
        summary["logit_children_margeff"] = float("nan")

    # Save summary for inspection
    out = pd.Series(summary)
    out.to_csv("analysis_summary.csv")

    # Determine scalar conclusion
    # Negative coefficients/marginal effects indicate children associated with fewer affairs.
    # Scale strength based on effect sizes and p-values.
    effect = 0.0
    weight = 0.0

    # Use standardized effect from mean difference
    if not np.isnan(summary["mean_affairs_diff_yes_minus_no"]):
        effect += -summary["mean_affairs_diff_yes_minus_no"]  # negative diff supports decrease
        weight += 1.0

    if not np.isnan(summary["any_affair_diff_yes_minus_no"]):
        effect += -summary["any_affair_diff_yes_minus_no"] * 10.0
        weight += 1.0

    if not np.isnan(summary["ols_children_coef"]):
        effect += -summary["ols_children_coef"]
        weight += 1.0

    if not np.isnan(summary["logit_children_margeff"]):
        effect += -summary["logit_children_margeff"] * 10.0
        weight += 1.0

    avg_effect = effect / max(weight, 1.0)

    # Convert to Likert scale. Typical effects are small; apply scaling.
    # Cap to [-100, 100].
    scalar = int(np.clip(avg_effect * 50.0, -100, 100))

    with open("conclusion.txt", "w") as f:
        f.write(str(scalar))


if __name__ == "__main__":
    main()
