import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    data_path = Path("affairs.csv")
    df = pd.read_csv(data_path)

    # Rename for clarity based on info.json
    df = df.rename(
        columns={
            "feature2": "affair_freq",  # coded frequency of extramarital intercourse
            "feature3": "gender",
            "feature4": "age",
            "feature5": "years_married",
            "feature6": "children",  # yes / no
            "feature7": "religiousness",
            "feature8": "education",
            "feature9": "occupation",
            "feature10": "marriage_rating",
        }
    )

    # Binary indicator of any affair in the past year
    df["has_affair"] = (df["affair_freq"] > 0).astype(int)
    df["children_bin"] = (df["children"] == "yes").astype(int)

    # Basic group summaries
    summary = {}

    group = df.groupby("children")
    summary["n_by_children"] = group.size().to_dict()
    summary["mean_affair_freq_by_children"] = group["affair_freq"].mean().to_dict()
    summary["prop_any_affair_by_children"] = group["has_affair"].mean().to_dict()

    # Two-sample t-test on affair frequency by children
    freq_children = df.loc[df["children_bin"] == 1, "affair_freq"]
    freq_no_children = df.loc[df["children_bin"] == 0, "affair_freq"]

    from scipy import stats

    t_stat, p_val_ttest = stats.ttest_ind(
        freq_children, freq_no_children, equal_var=False
    )
    summary["ttest_affair_freq_children_vs_no_children"] = {
        "t_stat": float(t_stat),
        "p_value": float(p_val_ttest),
    }

    # Logistic regression: any affair ~ children (+ controls)
    logit_simple = smf.logit("has_affair ~ children_bin", data=df).fit(disp=False)
    logit_controls = smf.logit(
        "has_affair ~ children_bin + age + years_married + religiousness + education + occupation + marriage_rating",
        data=df,
    ).fit(disp=False)

    def extract_logit_results(model):
        params = model.params.to_dict()
        pvalues = model.pvalues.to_dict()
        conf_int = model.conf_int()
        conf = {
            name: {"lower": float(conf_int.loc[name, 0]), "upper": float(conf_int.loc[name, 1])}
            for name in conf_int.index
        }
        return {
            "params": {k: float(v) for k, v in params.items()},
            "pvalues": {k: float(v) for k, v in pvalues.items()},
            "conf_int": conf,
        }

    summary["logit_simple_children_has_affair"] = extract_logit_results(logit_simple)
    summary["logit_controls_children_has_affair"] = extract_logit_results(logit_controls)

    # Poisson regression on frequency as count outcome
    poisson_simple = smf.poisson("affair_freq ~ children_bin", data=df).fit(disp=False)
    poisson_controls = smf.poisson(
        "affair_freq ~ children_bin + age + years_married + religiousness + education + occupation + marriage_rating",
        data=df,
    ).fit(disp=False)

    def extract_glm_results(model):
        params = model.params.to_dict()
        pvalues = model.pvalues.to_dict()
        return {
            "params": {k: float(v) for k, v in params.items()},
            "pvalues": {k: float(v) for k, v in pvalues.items()},
        }

    summary["poisson_simple_children_affair_freq"] = extract_glm_results(poisson_simple)
    summary["poisson_controls_children_affair_freq"] = extract_glm_results(poisson_controls)

    # Save a machine-readable summary for inspection
    with open("analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

