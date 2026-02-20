import pandas as pd
import statsmodels.api as sm
from pathlib import Path


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_variables(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    # According to info.json metadata:
    # - column "age" encodes frequency of extramarital intercourse (0,1,2,3,7,12)
    # - column "religiousness" is actually a yes/no factor for presence of children
    # The column names are shuffled, so we rely on metadata rather than names.
    data["affairs_freq"] = data["age"]
    data["any_affair"] = (data["affairs_freq"] > 0).astype(int)

    # Children in marriage: "yes"/"no"
    data["has_children"] = data["religiousness"].map({"yes": 1, "no": 0})

    # Additional covariates based on metadata mapping
    # - "occupation" column encodes age in years (17.5–57)
    # - "children" column encodes years married
    # - "rating" column encodes religiousness (1–5)
    # - "yearsmarried" column encodes education (9–20)
    # - "rownames" column encodes occupation category (1–7)
    # - "affairs" column encodes self rating of marriage (1–5)
    data["age_years"] = data["occupation"]
    data["years_married"] = data["children"]
    data["religiosity"] = data["rating"]
    data["education"] = data["yearsmarried"]
    data["occupation_cat"] = data["rownames"]
    data["marriage_rating"] = data["affairs"]

    # Encode gender as binary indicator for convenience (female=1, male=0)
    if "gender" in data.columns:
        data["female"] = (data["gender"] == "female").astype(int)

    return data


def descriptive_comparison(data: pd.DataFrame) -> None:
    print("=== Descriptive comparison of affairs by children status ===")
    tab = (
        data.groupby("has_children")["any_affair"]
        .agg(["mean", "sum", "count"])
        .rename(index={0: "no_children", 1: "has_children"})
    )
    print(tab)

    # Average affair frequency (treating codes as approximate counts)
    freq_tab = (
        data.groupby("has_children")["affairs_freq"]
        .agg(["mean", "median"])
        .rename(index={0: "no_children", 1: "has_children"})
    )
    print("\n=== Average extramarital intercourse frequency (coded) ===")
    print(freq_tab)


def logistic_regression(data: pd.DataFrame) -> None:
    print("\n=== Logistic regression: any_affair ~ has_children + controls ===")

    covariates = [
        "has_children",
        "age_years",
        "years_married",
        "religiosity",
        "education",
        "occupation_cat",
        "marriage_rating",
        "female",
    ]
    X = data[covariates].astype(float)
    X = sm.add_constant(X)
    y = data["any_affair"].astype(float)

    model = sm.Logit(y, X)
    result = model.fit(disp=False)
    print(result.summary())

    # Compute predicted probabilities for a "typical" individual with/without children
    mean_vals = X.mean()
    for has_children in [0, 1]:
        scenario = mean_vals.copy()
        scenario["has_children"] = has_children
        scenario_df = pd.DataFrame([scenario])
        prob = float(result.predict(scenario_df)[0])
        label = "with_children" if has_children == 1 else "without_children"
        print(f"Predicted probability of any affair ({label}): {prob:.3f}")


def main() -> None:
    csv_path = Path("affairs.csv")
    df_raw = load_data(csv_path)
    data = prepare_variables(df_raw)

    descriptive_comparison(data)
    logistic_regression(data)


if __name__ == "__main__":
    main()
