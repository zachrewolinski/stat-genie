import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("affairs.csv")

    df["has_children"] = (df["feature6"].str.lower() == "yes").astype(int)
    df["has_affair"] = (df["feature2"] > 0).astype(int)
    df["is_female"] = (df["feature3"].str.lower() == "female").astype(int)

    X = df[
        [
            "has_children",
            "is_female",
            "feature4",  # age
            "feature5",  # years married
            "feature7",
            "feature8",
            "feature9",
            "feature10",
        ]
    ]
    X = sm.add_constant(X)
    y = df["has_affair"]

    model = sm.Logit(y, X).fit(disp=False)

    # Print a compact summary line for the children coefficient
    coef = float(model.params["has_children"])
    pvalue = float(model.pvalues["has_children"])
    print(f"has_children coef={coef:.4f}, pvalue={pvalue:.4f}")


if __name__ == "__main__":
    main()
