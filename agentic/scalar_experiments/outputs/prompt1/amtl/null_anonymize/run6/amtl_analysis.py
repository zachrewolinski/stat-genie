import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.copy()

    df["tooth_class"] = df["feature1"].astype("category")
    df["specimen_id"] = df["feature2"].astype("category")
    df["missing"] = pd.to_numeric(df["feature3"], errors="coerce")
    df["sockets"] = pd.to_numeric(df["feature4"], errors="coerce")
    df["age"] = pd.to_numeric(df["feature5"], errors="coerce")
    df["age_uncertainty"] = pd.to_numeric(df["feature6"], errors="coerce")
    df["sex_estimate"] = pd.to_numeric(df["feature7"], errors="coerce")
    df["genus"] = df["feature8"].astype("category")
    df["region"] = df["feature9"].astype("category")

    df = df.dropna(subset=["missing", "sockets", "age", "sex_estimate"])

    df = df[df["sockets"] > 0]

    df = df[df["missing"] >= 0]

    df = df[df["missing"] <= df["sockets"]]

    return df


def fit_model(df: pd.DataFrame):
    df = df.copy()

    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    tooth_class_dummies = pd.get_dummies(df["tooth_class"], prefix="tooth", drop_first=True)
    X = pd.concat(
        [
            pd.Series(1.0, index=df.index, name="intercept"),
            df["is_human"],
            df["age"],
            df["sex_estimate"],
            tooth_class_dummies,
        ],
        axis=1,
    )

    successes = df["missing"]
    failures = df["sockets"] - df["missing"]
    endog = np.column_stack((successes, failures))

    model = sm.GLM(endog, X, family=sm.families.Binomial())
    result = model.fit()
    return result


def interpret_result(result) -> dict:
    coef = result.params["is_human"]
    p_value = result.pvalues["is_human"]

    yes = coef > 0 and p_value < 0.05
    response = "Yes" if yes else "No"

    summary_lines = []
    summary_lines.append(
        "Binomial regression model of AMTL (missing teeth out of observable sockets) "
        "was fit with predictors: human vs. non-human, age at death, sex estimate, and tooth class."
    )
    summary_lines.append(
        f"The coefficient for the human indicator (Homo sapiens vs. non-human primates) "
        f"was {coef:.3f} with p-value {p_value:.4g}."
    )
    if yes:
        summary_lines.append(
            "This positive and statistically significant coefficient indicates that, "
            "after adjusting for age, sex, and tooth class, modern humans have higher "
            "frequencies of antemortem tooth loss than the non-human primate genera "
            "(Pan, Pongo, Papio)."
        )
    else:
        summary_lines.append(
            "This coefficient is not positive and statistically significant, so the data "
            "do not provide evidence that modern humans have higher AMTL frequencies than "
            "the non-human primate genera after adjusting for age, sex, and tooth class."
        )

    return {"response": response, "explanation": " ".join(summary_lines)}


def main():
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "amtl.csv"

    df = load_data(csv_path)
    if df.empty:
        conclusion = {
            "response": "No",
            "explanation": "After filtering invalid records, no data remained to evaluate differences in AMTL between humans and non-human primates.",
        }
    else:
        try:
            result = fit_model(df)
            conclusion = interpret_result(result)
        except Exception as exc:
            conclusion = {
                "response": "No",
                "explanation": (
                    "An error occurred while fitting the binomial regression model "
                    "to compare AMTL frequencies between humans and non-human primates "
                    f"after adjusting for age, sex, and tooth class: {exc!r}"
                ),
            }

    conclusion_path = base_dir / "conclusion.txt"
    with conclusion_path.open("w", encoding="utf-8") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()

