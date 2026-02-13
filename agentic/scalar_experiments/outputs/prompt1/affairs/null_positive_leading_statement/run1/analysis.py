import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Define a binary indicator for any extramarital affair in the past year.
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Basic descriptive comparison by children status.
    group = df.groupby("children", observed=True)
    mean_affairs = group["affairs"].mean()
    prop_any_affair = group["any_affair"].mean()

    # Unadjusted logistic regression: probability of any affair ~ children.
    # children is treated as a categorical variable with "no" as the reference.
    logit_formula_simple = "any_affair ~ C(children)"
    simple_model = smf.logit(logit_formula_simple, data=df).fit(disp=False)
    simple_params = simple_model.params
    simple_pvalues = simple_model.pvalues

    # Extract coefficient and p-value for having children (vs no children).
    # With C(children), statsmodels creates a term like C(children)[T.yes].
    child_term = "C(children)[T.yes]"
    simple_coef = float(simple_params.get(child_term, np.nan))
    simple_p = float(simple_pvalues.get(child_term, np.nan))
    simple_or = float(np.exp(simple_coef)) if np.isfinite(simple_coef) else np.nan

    # Adjusted logistic regression adding key covariates used in the literature.
    logit_formula_adj = (
        "any_affair ~ C(children) + C(gender) + age + yearsmarried + "
        "religiousness + education + occupation + rating"
    )
    try:
        adj_model = smf.logit(logit_formula_adj, data=df).fit(disp=False)
        adj_params = adj_model.params
        adj_pvalues = adj_model.pvalues
        adj_coef = float(adj_params.get(child_term, np.nan))
        adj_p = float(adj_pvalues.get(child_term, np.nan))
        adj_or = float(np.exp(adj_coef)) if np.isfinite(adj_coef) else np.nan
    except Exception:
        # If the adjusted model fails to converge for any reason,
        # fall back to the simple model only.
        adj_coef = np.nan
        adj_p = np.nan
        adj_or = np.nan

    # Decide on the answer:
    # We answer "Yes" only if having children is associated with
    # a statistically significant DECREASE in the probability of
    # any extramarital affair (odds ratio < 1 and p < 0.05) in
    # both the simple and adjusted models (when available).
    def is_protective(coef: float, pval: float) -> bool:
        return np.isfinite(coef) and coef < 0 and np.isfinite(pval) and pval < 0.05

    simple_protective = is_protective(simple_coef, simple_p)
    adj_protective = is_protective(adj_coef, adj_p) if np.isfinite(adj_coef) else True

    if simple_protective and adj_protective:
        response = "Yes"
    else:
        response = "No"

    # Build a concise explanation with concrete numbers.
    # Descriptive statistics
    mean_affairs_children_yes = float(mean_affairs.get("yes", np.nan))
    mean_affairs_children_no = float(mean_affairs.get("no", np.nan))
    prop_any_affair_children_yes = float(prop_any_affair.get("yes", np.nan))
    prop_any_affair_children_no = float(prop_any_affair.get("no", np.nan))

    explanation = (
        "I examined the Psychology Today marital affairs dataset (601 married individuals) "
        "to test whether having children decreases engagement in extramarital affairs. "
        f"Descriptively, the mean number of affairs during the past year was "
        f"{mean_affairs_children_yes:.2f} for respondents with children and "
        f"{mean_affairs_children_no:.2f} for those without children. "
        f"The proportion who reported at least one affair was "
        f"{prop_any_affair_children_yes:.2%} with children versus "
        f"{prop_any_affair_children_no:.2%} without children. "
        "I then fit a logistic regression predicting any affair (yes/no) from children status. "
        f"In the simple model, the log-odds coefficient for having children was "
        f"{simple_coef:.3f} (odds ratio {simple_or:.2f}, p-value {simple_p:.3f}). "
    )

    if np.isfinite(adj_coef):
        explanation += (
            "Next, I fit an adjusted logistic model including gender, age, years married, "
            "religiousness, education, occupation, and self-rated marital happiness. "
            f"In this adjusted model, the coefficient for having children was "
            f"{adj_coef:.3f} (odds ratio {adj_or:.2f}, p-value {adj_p:.3f}). "
        )

    if response == "Yes":
        explanation += (
            "In both models, having children is associated with a statistically significant "
            "reduction in the probability of any extramarital affair (odds ratio below 1 "
            "with p < 0.05), providing evidence that, in this sample, having children does "
            "indeed decrease engagement in extramarital affairs."
        )
    else:
        explanation += (
            "These results do not show a statistically significant protective effect of "
            "having children against extramarital affairs (the children coefficient is not "
            "both negative and significant at the 5% level across models). "
            "Given the descriptive differences and regression estimates, the data do not "
            "provide strong evidence that having children decreases engagement in "
            "extramarital affairs; at most, any difference appears small and statistically "
            "uncertain in this sample."
        )

    conclusion = {"response": response, "explanation": explanation}

    conclusion_path = Path("conclusion.txt")
    conclusion_path.write_text(json.dumps(conclusion), encoding="utf-8")


if __name__ == "__main__":
    main()

