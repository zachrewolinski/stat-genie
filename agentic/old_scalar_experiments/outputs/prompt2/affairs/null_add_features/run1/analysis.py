import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    base_dir = Path(__file__).parent

    # Load metadata / research question
    info_path = base_dir / "info.json"
    with info_path.open("r") as f:
        info = json.load(f)

    research_question = info.get("research_questions", [""])[0]

    # Load dataset
    data_path = base_dir / "affairs.csv"
    df = pd.read_csv(data_path)

    # Define binary indicator of any extramarital affair
    df["has_affair"] = (df["affairs"] > 0).astype(int)

    # Basic group-level summaries by children status
    group_summary = (
        df.groupby("children")
        .agg(
            mean_affairs=("affairs", "mean"),
            prop_any_affair=("has_affair", "mean"),
            n=("affairs", "size"),
        )
        .reset_index()
    )

    print("Research question:", research_question)
    print("\nGroup summary by children status:")
    print(group_summary.to_string(index=False))

    # Logistic regression: any affair ~ children + covariates
    df["children_flag"] = (df["children"] == "yes").astype(int)

    # Select available covariates (only those actually present in the CSV)
    candidate_covariates = [
        "age",
        "yearsmarried",
        "religiousness",
        "education",
        "occupation",
        "rating",
        "m_focal",
        "num_words",
        "income",
    ]
    covariates = [c for c in candidate_covariates if c in df.columns]

    y = df["has_affair"]
    X_cols = ["children_flag"] + covariates
    X = df[X_cols].copy()
    X = sm.add_constant(X, has_constant="add")

    logit_model = sm.Logit(y, X).fit(disp=False)

    print("\nLogistic regression results: has_affair ~ children + covariates")
    print(logit_model.summary())

    # Extract effect of children on the odds of any affair
    coef_children = float(logit_model.params["children_flag"])
    conf_int_children = logit_model.conf_int().loc["children_flag"].tolist()
    pval_children = float(logit_model.pvalues["children_flag"])
    odds_ratio_children = float(np.exp(coef_children))

    print("\nEffect of children_flag on odds of any affair:")
    print(f"  Coefficient (log-odds): {coef_children:.4f}")
    print(
        "  95% CI (log-odds): "
        f"[{conf_int_children[0]:.4f}, {conf_int_children[1]:.4f}]"
    )
    print(f"  Odds ratio: {odds_ratio_children:.4f}")
    print(f"  p-value: {pval_children:.4g}")


if __name__ == "__main__":
    main()

