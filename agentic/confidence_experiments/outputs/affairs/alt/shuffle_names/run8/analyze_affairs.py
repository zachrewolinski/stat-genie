import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Recode variables to their semantic meanings based on info.json
    df["affair_freq"] = df["age"]  # frequency of extramarital intercourse
    df["has_affair"] = (df["affair_freq"] > 0).astype(int)
    df["has_children"] = df["religiousness"].map({"yes": 1, "no": 0})

    # Additional covariates with clearer semantic names
    df["age_years"] = df["occupation"]
    df["years_married"] = df["children"]
    df["religiousness_level"] = df["rating"]
    df["education_level"] = df["yearsmarried"]
    df["occupation_code"] = df["rownames"]
    df["marriage_rating"] = df["affairs"]

    df = df.dropna(subset=["has_affair", "has_children"])

    # Basic descriptive statistics
    print("N observations:", len(df))
    print()

    by_children = df.groupby("has_children")["has_affair"].agg(["mean", "sum", "count"])
    print("Prevalence of any extramarital intercourse by children status:")
    print(by_children)
    print()

    freq_by_children = df.groupby("has_children")["affair_freq"].mean()
    print("Mean frequency of extramarital intercourse by children status:")
    print(freq_by_children)
    print()

    # Unadjusted logistic regression: any affair ~ children
    print("Unadjusted logistic regression: has_affair ~ has_children")
    model_unadj = smf.logit("has_affair ~ has_children", data=df).fit(disp=False)
    or_unadj = float(np.exp(model_unadj.params["has_children"]))
    ci_unadj = np.exp(model_unadj.conf_int().loc["has_children"])
    p_unadj = float(model_unadj.pvalues["has_children"])
    print(model_unadj.summary())
    print(f"Unadjusted OR (children vs no children): {or_unadj:.3f}")
    print(
        "95% CI for OR: "
        f"({ci_unadj[0]:.3f}, {ci_unadj[1]:.3f}), "
        f"p-value = {p_unadj:.4f}"
    )
    print()

    # Adjusted logistic regression including key covariates
    print(
        "Adjusted logistic regression: has_affair ~ has_children + "
        "age_years + years_married + C(gender) + religiousness_level + "
        "education_level + occupation_code + marriage_rating"
    )
    model_adj = smf.logit(
        "has_affair ~ has_children + age_years + years_married + "
        "C(gender) + religiousness_level + education_level + "
        "occupation_code + marriage_rating",
        data=df,
    ).fit(disp=False)

    or_adj = float(np.exp(model_adj.params["has_children"]))
    ci_adj = np.exp(model_adj.conf_int().loc["has_children"])
    p_adj = float(model_adj.pvalues["has_children"])
    print(model_adj.summary())
    print(f"Adjusted OR (children vs no children): {or_adj:.3f}")
    print(
        "95% CI for adjusted OR: "
        f"({ci_adj[0]:.3f}, {ci_adj[1]:.3f}), "
        f"p-value = {p_adj:.4f}"
    )


if __name__ == "__main__":
    main()

