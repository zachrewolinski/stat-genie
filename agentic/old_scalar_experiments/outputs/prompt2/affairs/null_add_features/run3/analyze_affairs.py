import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


DATA_PATH = Path("affairs.csv")


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    # Binary outcome: any extramarital affair in past year
    df["had_affair"] = (df["affairs"] > 0).astype(int)

    # Basic descriptive comparison by children status
    desc = (
        df.groupby("children")["had_affair"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "prop_with_affair"})
    )

    # Logistic regression of having an affair on children and key covariates
    # Use a simple specification focused on the classic affairs predictors.
    formula = (
        "had_affair ~ C(children) + C(gender) + age + yearsmarried "
        "+ religiousness + education + occupation + rating"
    )

    model = smf.logit(formula=formula, data=df).fit(disp=False)
    params = model.params
    pvalues = model.pvalues

    # Extract coefficient for children effect (indicator for children[T.yes])
    # Depending on baseline coding, statsmodels will produce either
    # C(children)[T.yes] or C(children)[T.no]. Handle both defensively.
    child_coef = None
    child_p = None

    if "C(children)[T.yes]" in params:
        child_coef = params["C(children)[T.yes]"]
        child_p = pvalues["C(children)[T.yes]"]
        child_level = "yes_vs_no"
    elif "C(children)[T.no]" in params:
        child_coef = params["C(children)[T.no]"]
        child_p = pvalues["C(children)[T.no]"]
        child_level = "no_vs_yes"
    else:
        child_level = "unknown"

    # Save a small JSON summary of key statistics so they can be inspected
    # separately from the final conclusion.
    summary = {
        "group_proportions": desc.reset_index().to_dict(orient="records"),
        "logit_children_coef": child_coef,
        "logit_children_pvalue": child_p,
        "logit_children_contrast": child_level,
        "n_obs": int(model.nobs),
    }

    Path("analysis_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

