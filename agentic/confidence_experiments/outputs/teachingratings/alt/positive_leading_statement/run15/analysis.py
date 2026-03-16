import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def main():
    df = pd.read_csv("teachingratings.csv")

    # Basic summaries
    summary = {
        "n": len(df),
        "beauty_mean": df["beauty"].mean(),
        "beauty_sd": df["beauty"].std(ddof=1),
        "eval_mean": df["eval"].mean(),
        "eval_sd": df["eval"].std(ddof=1),
        "corr_beauty_eval": df["beauty"].corr(df["eval"]),
    }

    # Simple bivariate model
    model_simple = smf.ols("eval ~ beauty", data=df).fit(cov_type="HC3")

    # Multivariate model with controls
    # Categorical controls: gender, minority, credits, division, native, tenure
    # Continuous controls: age, students, allstudents
    # Use log(1+students) and log(1+allstudents) to dampen skew
    df = df.copy()
    df["log_students"] = np.log1p(df["students"])
    df["log_allstudents"] = np.log1p(df["allstudents"])

    model_controls = smf.ols(
        "eval ~ beauty + age + C(gender) + C(minority) + C(credits) + "
        "C(division) + C(native) + C(tenure) + log_students + log_allstudents",
        data=df,
    ).fit(cov_type="HC3")

    def model_info(m):
        return {
            "coef_beauty": m.params.get("beauty"),
            "se_beauty": m.bse.get("beauty"),
            "p_beauty": m.pvalues.get("beauty"),
            "r2": m.rsquared,
        }

    results = {
        "summary": summary,
        "simple": model_info(model_simple),
        "controls": model_info(model_controls),
    }

    with open("analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
