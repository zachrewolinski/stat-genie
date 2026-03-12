import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Binary indicator for any extramarital affair in past year
    df["any_affair"] = (df["affairs"] > 0).astype(int)

    # Basic group summaries by children status
    summary = (
        df.groupby("children")
        .agg(
            n=("any_affair", "size"),
            any_affair_rate=("any_affair", "mean"),
            mean_affairs=("affairs", "mean"),
        )
        .reset_index()
    )

    print("Group summaries by children status:")
    print(summary.to_string(index=False))
    print()

    # Logistic regression: probability of any affair ~ children + controls
    # children is categorical yes/no; include as C(children)
    formula = (
        "any_affair ~ C(children) + age + yearsmarried + C(gender) + "
        "religiousness + education + occupation + rating"
    )

    logit_model = smf.logit(formula=formula, data=df).fit(disp=False)
    print("Logistic regression results (any_affair as outcome):")
    print(logit_model.summary())
    print()

    # Extract the coefficient and p-value for children effect.
    # With C(children), statsmodels will typically create C(children)[T.yes] if 'no' is baseline.
    children_term = None
    for term in logit_model.params.index:
        if term.startswith("C(children)[T."):
            children_term = term
            break

    if children_term is None:
        raise RuntimeError("Could not find children term in logistic regression.")

    beta_children = float(logit_model.params[children_term])
    p_children = float(logit_model.pvalues[children_term])

    print(f"Coefficient for {children_term}: {beta_children:.4f}, p-value={p_children:.4g}")

    # Compute marginal predicted probabilities for having any affair
    # for a typical individual with and without children.
    # Use the sample means of continuous covariates and the most common category for factors.
    covariates = ["age", "yearsmarried", "religiousness", "education", "occupation", "rating"]
    means = df[covariates].mean()

    # Most common category for gender
    common_gender = df["gender"].mode().iat[0]

    def make_row(children_value: str) -> pd.DataFrame:
        row = {**means.to_dict()}
        row["gender"] = common_gender
        row["children"] = children_value
        return pd.DataFrame([row])

    pred_no_children = float(
        logit_model.predict(make_row("no")).iloc[0]
    )
    pred_yes_children = float(
        logit_model.predict(make_row("yes")).iloc[0]
    )

    print(
        f"Predicted probability of any affair (typical profile): "
        f"no children={pred_no_children:.3f}, children={pred_yes_children:.3f}"
    )

    # Save intermediate numerical results for manual inspection if needed.
    results = {
        "group_summary": summary.to_dict(orient="list"),
        "logit_children_coefficient": beta_children,
        "logit_children_p_value": p_children,
        "pred_prob_no_children": pred_no_children,
        "pred_prob_children": pred_yes_children,
    }

    Path("analysis_results.json").write_text(json.dumps(results, indent=2))
    print("\nSaved detailed numerical results to analysis_results.json")


if __name__ == "__main__":
    main()

