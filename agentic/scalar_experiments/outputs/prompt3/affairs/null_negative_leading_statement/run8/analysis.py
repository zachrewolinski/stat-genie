import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Create a binary indicator for having any extramarital affair
    df["any_affair"] = (df["affairs"] > 0).astype(int)
    return df


def summarize_by_children(df: pd.DataFrame) -> dict:
    grouped = df.groupby("children")
    summary = grouped["affairs"].agg(["mean", "std", "count"]).to_dict(orient="index")
    prop_any = grouped["any_affair"].mean().to_dict()
    for k in summary:
        summary[k]["prop_any_affair"] = prop_any.get(k, np.nan)
    return summary


def run_logistic_regression(df: pd.DataFrame):
    """
    Logistic regression of any_affair on children, controlling for key covariates.
    """
    formula = (
        "any_affair ~ C(children) + age + yearsmarried + religiousness "
        "+ education + occupation + rating"
    )
    model = smf.logit(formula=formula, data=df).fit(disp=False)
    return model


def main():
    data_path = Path("affairs.csv")
    df = load_data(data_path)

    # Basic summaries
    summary = summarize_by_children(df)

    # Logistic regression
    logit_model = run_logistic_regression(df)
    params = logit_model.params
    conf_int = logit_model.conf_int()

    # Extract effect of children (yes vs no).
    # With C(children), the coefficient is typically C(children)[T.yes]
    coef_name_options = [c for c in params.index if "C(children)" in c]
    children_effect = None
    if coef_name_options:
        coef_name = coef_name_options[0]
        children_coef = params[coef_name]
        ci_low, ci_high = conf_int.loc[coef_name]
        odds_ratio = float(np.exp(children_coef))
        children_effect = {
            "coef_name": coef_name,
            "logit_coef": float(children_coef),
            "odds_ratio": odds_ratio,
            "ci_low": float(np.exp(ci_low)),
            "ci_high": float(np.exp(ci_high)),
            "p_value": float(logit_model.pvalues[coef_name]),
        }

    results = {
        "n_obs": int(df.shape[0]),
        "summary_by_children": summary,
        "children_effect_logit": children_effect,
    }

    # Print as JSON so we can inspect from the CLI.
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

