import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Outcome: any extramarital intercourse in past year
    df["any_affair"] = (df["feature2"] > 0).astype(int)

    # Basic group summaries by children status
    group = df.groupby("feature6")
    summary = {
        "mean_affair_frequency": group["feature2"].mean().to_dict(),
        "prop_with_any_affair": group["any_affair"].mean().to_dict(),
        "count": group.size().to_dict(),
    }

    # Logistic regression: any affair ~ children + controls
    # feature6 is 'yes'/'no' for children in marriage.
    # Controls: gender, age, years married, religiousness,
    # education, occupation, and self-rated marriage.
    formula = (
        "any_affair ~ C(feature6) + C(feature3) + feature4 + "
        "feature5 + feature7 + feature8 + feature9 + feature10"
    )
    logit_model = smf.logit(formula, data=df).fit(disp=False)
    logit_summary = str(logit_model.summary())

    # Save a lightweight JSON with key numerical results to inspect manually.
    results = {
        "group_summary": summary,
        "children_coef": logit_model.params.get("C(feature6)[T.yes]", None),
        "children_pvalue": logit_model.pvalues.get("C(feature6)[T.yes]", None),
    }

    Path("analysis_results.json").write_text(json.dumps(results, indent=2))
    Path("logit_summary.txt").write_text(logit_summary)


if __name__ == "__main__":
    main()

