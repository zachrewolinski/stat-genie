import json

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Binary indicator of any extramarital affair in the past year
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Basic group statistics by children status
    group = df.groupby("children")
    summary = {
        "n_by_children": group["has_affair"].count().to_dict(),
        "mean_affairs_by_children": group["affairs"].mean().to_dict(),
        "prop_any_affair_by_children": group["has_affair"].mean().to_dict(),
    }

    # Logistic regression of any affair on children and controls
    # children and gender are treated as categorical; others numeric.
    formula = (
        "has_affair ~ C(children) + C(gender) + age + yearsmarried "
        "+ religiousness + education + occupation + rating"
    )
    logit_model = smf.logit(formula=formula, data=df).fit(disp=False)
    params = logit_model.params.to_dict()
    pvalues = logit_model.pvalues.to_dict()

    effect_children_yes = params.get("C(children)[T.yes]", float("nan"))
    p_children_yes = pvalues.get("C(children)[T.yes]", float("nan"))

    summary["logit_coef_children_yes"] = effect_children_yes
    summary["logit_p_children_yes"] = p_children_yes

    # Print a compact JSON summary so the calling environment can inspect it
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

