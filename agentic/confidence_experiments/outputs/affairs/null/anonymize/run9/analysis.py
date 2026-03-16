import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Rename for clarity
    df = df.rename(
        columns={
            "feature2": "affair_freq",
            "feature3": "gender",
            "feature4": "age",
            "feature5": "years_married",
            "feature6": "children_raw",
            "feature7": "religiousness",
            "feature8": "education",
            "feature9": "occupation",
            "feature10": "marriage_rating",
        }
    )

    # Children indicator: 1 if there are children in the marriage, 0 otherwise
    df["children"] = (df["children_raw"].str.lower() == "yes").astype(int)

    # Any affair indicator: 1 if engaged in any extramarital intercourse in past year
    df["any_affair"] = (df["affair_freq"] > 0).astype(int)

    # Basic group summaries
    group_stats = (
        df.groupby("children")
        .agg(
            mean_affair_freq=("affair_freq", "mean"),
            median_affair_freq=("affair_freq", "median"),
            prop_any_affair=("any_affair", "mean"),
            n=("any_affair", "size"),
        )
        .reset_index()
    )

    # Two-sample t-test for mean affair frequency
    freq_children = df.loc[df["children"] == 1, "affair_freq"]
    freq_no_children = df.loc[df["children"] == 0, "affair_freq"]
    t_stat, t_pvalue = stats.ttest_ind(
        freq_children, freq_no_children, equal_var=False
    )

    # Difference in proportions for any affair
    count_affair = df.groupby("children")["any_affair"].sum()
    n_obs = df.groupby("children")["any_affair"].count()
    # children=0 is index 0, children=1 is index 1 after sorting by index
    count = np.array([count_affair.loc[0], count_affair.loc[1]])
    nobs = np.array([n_obs.loc[0], n_obs.loc[1]])
    stat_prop, pvalue_prop = sm.stats.proportions_ztest(count, nobs)

    # Logistic regression: any affair on children only
    logit_children = smf.logit("any_affair ~ children", data=df).fit(disp=False)
    params_children = logit_children.params
    conf_children = logit_children.conf_int()
    odds_ratio_children = np.exp(params_children["children"])
    ci_low_children = np.exp(conf_children.loc["children", 0])
    ci_high_children = np.exp(conf_children.loc["children", 1])

    # Logistic regression with basic controls
    logit_full = smf.logit(
        "any_affair ~ children + age + years_married + C(gender)"
        " + religiousness + education + occupation + marriage_rating",
        data=df,
    ).fit(disp=False)
    params_full = logit_full.params
    conf_full = logit_full.conf_int()
    odds_ratio_children_full = np.exp(params_full["children"])
    ci_low_children_full = np.exp(conf_full.loc["children", 0])
    ci_high_children_full = np.exp(conf_full.loc["children", 1])

    results = {
        "group_stats": group_stats.to_dict(orient="records"),
        "t_test": {
            "t_stat": float(t_stat),
            "p_value": float(t_pvalue),
        },
        "prop_test": {
            "z_stat": float(stat_prop),
            "p_value": float(pvalue_prop),
            "counts": count.tolist(),
            "nobs": nobs.tolist(),
        },
        "logit_children": {
            "params": params_children.to_dict(),
            "odds_ratio_children": float(odds_ratio_children),
            "ci_children": [float(ci_low_children), float(ci_high_children)],
            "p_value_children": float(logit_children.pvalues["children"]),
        },
        "logit_full": {
            "params": params_full.to_dict(),
            "odds_ratio_children": float(odds_ratio_children_full),
            "ci_children": [
                float(ci_low_children_full),
                float(ci_high_children_full),
            ],
            "p_value_children": float(logit_full.pvalues["children"]),
        },
    }

    # Write a machine-readable JSON summary; human explanation will be built separately.
    Path("analysis_results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

