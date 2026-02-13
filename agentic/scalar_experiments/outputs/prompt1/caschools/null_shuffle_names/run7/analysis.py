import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # According to info.json, "english" is total enrollment and "students" is number of teachers.
    enrollment = df["english"].astype(float)
    teachers = df["students"].astype(float)

    # Exclude any nonpositive or missing teacher counts to avoid invalid ratios.
    mask = teachers > 0
    enrollment = enrollment[mask]
    teachers = teachers[mask]
    trimmed = df.loc[mask].copy()

    ratio = enrollment / teachers
    trimmed["stratio"] = ratio

    # Academic performance: mean of reading and math scores
    read_score = trimmed["district"].astype(float)
    math_score = trimmed["expenditure"].astype(float)
    performance = (read_score + math_score) / 2.0
    trimmed["performance"] = performance

    # Simple correlation
    corr = np.corrcoef(trimmed["stratio"], trimmed["performance"])[0, 1]

    # Simple linear regression: performance ~ student-teacher ratio
    X = sm.add_constant(trimmed["stratio"])
    y = trimmed["performance"]
    model = sm.OLS(y, X).fit()

    slope = model.params["stratio"]
    p_value = model.pvalues["stratio"]

    print("Number of districts used:", len(trimmed))
    print("Mean student-teacher ratio:", trimmed["stratio"].mean())
    print("Std student-teacher ratio:", trimmed["stratio"].std())
    print("Correlation (ratio, performance):", corr)
    print("OLS slope on ratio:", slope)
    print("OLS p-value for ratio:", p_value)

    # Also run a multiple regression controlling for key demographics.
    controls = ["income", "school", "computer", "rownames", "grades"]
    control_data = trimmed[controls].astype(float)
    Xc = sm.add_constant(pd.concat([trimmed["stratio"], control_data], axis=1))
    model_controls = sm.OLS(y, Xc).fit()

    slope_c = model_controls.params["stratio"]
    p_value_c = model_controls.pvalues["stratio"]

    print("\nWith controls (income, poverty, lunch, English learners, expenditure per student):")
    print("OLS slope on ratio (controlled):", slope_c)
    print("OLS p-value for ratio (controlled):", p_value_c)


if __name__ == "__main__":
    main()

