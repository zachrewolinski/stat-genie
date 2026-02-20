import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    df = pd.read_csv("affairs.csv")

    # Map scrambled column names to their semantic meaning based on info.json.
    df = df.copy()
    df["affair_freq"] = df["age"]  # how often engaged in extramarital intercourse (0 = none)
    df["any_affair"] = (df["affair_freq"] > 0).astype(int)
    df["has_children"] = df["religiousness"].map({"yes": 1, "no": 0})

    # Basic sanity checks
    print("N =", len(df))
    print(df[["affair_freq", "any_affair", "has_children"]].head())

    # Descriptive statistics by child status
    grouped = (
        df.groupby("has_children")
        .agg(
            mean_affair_freq=("affair_freq", "mean"),
            prop_any_affair=("any_affair", "mean"),
            n=("any_affair", "size"),
        )
        .reset_index()
    )
    print("\nDescriptive stats by has_children (1=yes, 0=no):")
    print(grouped)

    # Logistic regression of any_affair on has_children only
    logit_model = smf.logit("any_affair ~ has_children", data=df).fit(disp=False)
    params = logit_model.params
    conf = logit_model.conf_int()

    coef = params["has_children"]
    conf_low, conf_high = conf.loc["has_children"]
    odds_ratio = float(np.exp(coef))
    or_low = float(np.exp(conf_low))
    or_high = float(np.exp(conf_high))

    print("\nLogistic regression: any_affair ~ has_children")
    print(logit_model.summary())
    print(
        f"\nCoefficient for has_children: {coef:.3f} "
        f"(95% CI [{conf_low:.3f}, {conf_high:.3f}])"
    )
    print(
        f"Odds ratio for has_children: {odds_ratio:.3f} "
        f"(95% CI [{or_low:.3f}, {or_high:.3f}])"
    )

    # Covariate-adjusted model including other relevant factors.
    # occupation ~ age band, children ~ years married, rating ~ religiosity,
    # yearsmarried ~ education, rownames ~ occupation code, affairs ~ marriage rating.
    adj_formula = (
        "any_affair ~ has_children + C(gender) + occupation + children + "
        "rating + yearsmarried + rownames + affairs"
    )
    logit_adj = smf.logit(adj_formula, data=df).fit(disp=False)
    adj_params = logit_adj.params
    adj_conf = logit_adj.conf_int()
    adj_coef = adj_params["has_children"]
    adj_low, adj_high = adj_conf.loc["has_children"]
    adj_or = float(np.exp(adj_coef))
    adj_or_low = float(np.exp(adj_low))
    adj_or_high = float(np.exp(adj_high))

    print("\nAdjusted logistic regression:")
    print(logit_adj.summary())
    print(
        f"\nAdjusted coefficient for has_children: {adj_coef:.3f} "
        f"(95% CI [{adj_low:.3f}, {adj_high:.3f}])"
    )
    print(
        f"Adjusted odds ratio for has_children: {adj_or:.3f} "
        f"(95% CI [{adj_or_low:.3f}, {adj_or_high:.3f}])"
    )


if __name__ == "__main__":
    main()
