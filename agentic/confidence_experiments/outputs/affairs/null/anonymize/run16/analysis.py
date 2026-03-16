import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def main() -> None:
    data_path = Path("affairs.csv")
    if not data_path.exists():
        raise FileNotFoundError("affairs.csv not found in current directory.")

    df = pd.read_csv(data_path)

    # Encode variables of interest
    df["has_children"] = (df["feature6"] == "yes").astype(int)
    df["affair_any"] = (df["feature2"] > 0).astype(int)

    # Basic group summaries for affairs frequency and any affair
    group_means = df.groupby("has_children")["feature2"].mean()
    group_medians = df.groupby("has_children")["feature2"].median()
    group_props = df.groupby("has_children")["affair_any"].mean()

    # Nonparametric test for distribution of affair frequency
    no_children = df.loc[df["has_children"] == 0, "feature2"]
    yes_children = df.loc[df["has_children"] == 1, "feature2"]
    u_stat, p_mwu = stats.mannwhitneyu(
        no_children, yes_children, alternative="two-sided"
    )

    # Chi-square test for any affair vs. children
    contingency = pd.crosstab(df["has_children"], df["affair_any"])
    chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency)

    # Logistic regression for any affair, adjusting for key covariates
    # feature4: age, feature5: years married, feature7: religiousness,
    # feature8: education, feature9: occupation, feature3: gender (categorical)
    logit_model = smf.logit(
        "affair_any ~ has_children + feature4 + feature5 + feature7 + feature8 + feature9 + C(feature3)",
        data=df,
    ).fit(disp=False)

    params = logit_model.params
    pvalues = logit_model.pvalues
    odds_ratios = params.apply(np.exp)

    results = {
        "group_means_feature2": group_means.to_dict(),
        "group_medians_feature2": group_medians.to_dict(),
        "group_props_affair_any": group_props.to_dict(),
        "mannwhitneyu_p": float(p_mwu),
        "chi2_p": float(p_chi2),
        "logit_children_coef": float(params["has_children"]),
        "logit_children_p": float(pvalues["has_children"]),
        "logit_children_odds_ratio": float(odds_ratios["has_children"]),
        "logit_summary": str(logit_model.summary()),
    }

    # Print results as JSON so they are easy to inspect from the CLI
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

