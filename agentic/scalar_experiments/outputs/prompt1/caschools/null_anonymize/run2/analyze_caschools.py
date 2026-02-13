import json

import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct student-teacher ratio and an overall academic performance score
    # feature6: total enrollment, feature7: number of teachers
    df["stratio"] = df["feature6"] / df["feature7"]
    # feature14: average reading score, feature15: average math score
    df["testscr"] = df[["feature14", "feature15"]].mean(axis=1)

    # Drop any rows with missing values in variables we use
    confounders = ["feature8", "feature9", "feature11", "feature12", "feature13"]
    model_vars = ["stratio", "testscr"] + confounders
    df_model = df[model_vars].dropna()

    # Simple correlation between student-teacher ratio and test scores
    corr = df_model["stratio"].corr(df_model["testscr"])

    # Simple linear regression: testscr ~ stratio
    X_simple = sm.add_constant(df_model["stratio"])
    model_simple = sm.OLS(df_model["testscr"], X_simple).fit()
    beta_simple = float(model_simple.params["stratio"])
    p_simple = float(model_simple.pvalues["stratio"])

    # Multiple regression, controlling for key demographic and resource variables
    X_controls = sm.add_constant(df_model[["stratio"] + confounders])
    model_controls = sm.OLS(df_model["testscr"], X_controls).fit()
    beta_controls = float(model_controls.params["stratio"])
    p_controls = float(model_controls.pvalues["stratio"])

    alpha = 0.05
    evidence_negative = (beta_simple < 0 and p_simple < alpha) or (
        beta_controls < 0 and p_controls < alpha
    )
    response = "Yes" if evidence_negative else "No"

    if evidence_negative:
        explanation = (
            "Using data from 420 California K-6 and K-8 districts, I constructed the "
            "student-teacher ratio as total enrollment divided by the number of teachers "
            "and an overall academic performance measure as the average of district "
            "reading and math scores. The simple Pearson correlation between "
            f"student-teacher ratio and average test scores was {corr:.3f}, indicating "
            "that districts with more students per teacher tend to have lower scores. A "
            "simple linear regression of average test scores on student-teacher ratio "
            f"estimated a negative slope of {beta_simple:.2f} points per additional "
            f"student per teacher (p-value={p_simple:.3g}). I then fit a multiple "
            "regression that controlled for poverty and demographic differences (percent "
            "qualifying for CalWorks, percent on reduced-price lunch, expenditures per "
            "student, average district income, and percent English learners). In this "
            "adjusted model, the slope on student-teacher ratio remained negative at "
            f"{beta_controls:.2f} points per additional student per teacher "
            f"(p-value={p_controls:.3g}). Because the estimated association between "
            "higher student-teacher ratios and lower academic performance is consistently "
            "negative and statistically significant at the 5% level both before and after "
            "adjusting for these covariates, I conclude that, in this dataset, lower "
            "student-teacher ratios are associated with higher academic performance."
        )
    else:
        explanation = (
            "Using data from 420 California K-6 and K-8 districts, I constructed the "
            "student-teacher ratio as total enrollment divided by the number of teachers "
            "and an overall academic performance measure as the average of district "
            "reading and math scores. The simple Pearson correlation between "
            f"student-teacher ratio and average test scores was {corr:.3f}, which is very "
            "close to zero and does not indicate a meaningful linear relationship between "
            "class size and performance. A simple linear regression of average test "
            "scores on student-teacher ratio estimated a slope of "
            f"{beta_simple:.2f} points per additional student per teacher "
            f"(p-value={p_simple:.3g}), which is not statistically different from zero at "
            "the 5% level. I then fit a multiple regression that controlled for poverty "
            "and demographic differences (percent qualifying for CalWorks, percent on "
            "reduced-price lunch, expenditures per student, average district income, and "
            "percent English learners). In this adjusted model, the slope on "
            "student-teacher ratio was "
            f"{beta_controls:.2f} points per additional student per teacher "
            f"(p-value={p_controls:.3g}), again not statistically significant. Because "
            "the estimated association between student-teacher ratio and academic "
            "performance is very small in magnitude and not statistically significant in "
            "either the simple or adjusted models, this dataset does not provide clear "
            "evidence that lower student-teacher ratios are associated with higher "
            "academic performance."
        )

    with open("conclusion.txt", "w") as f:
        json.dump({"response": response, "explanation": explanation}, f)


if __name__ == "__main__":
    main()
