import pandas as pd
import statsmodels.api as sm


def main() -> None:
    df = pd.read_csv("crofoot.csv")

    # Relative group size: positive when focal group is larger.
    df["rel_size"] = df["n_focal"] - df["n_other"]
    df["focal_larger"] = (df["rel_size"] > 0).astype(int)

    # Contest location: positive when focal group is closer to its home range center.
    df["rel_dist"] = df["dist_other"] - df["dist_focal"]
    df["focal_closer"] = (df["rel_dist"] > 0).astype(int)

    print("Number of contests:", len(df))
    print("Overall focal win rate:", df["win"].mean())
    print()

    # Simple win rates by relative group size.
    print("Win rate by whether focal group is larger:")
    print(
        df.groupby("focal_larger")["win"]
        .agg(["mean", "count"])
        .rename(index={0: "focal_not_larger", 1: "focal_larger"})
    )
    print()

    # Simple win rates by contest location advantage.
    print("Win rate by whether focal group is closer to home range center:")
    print(
        df.groupby("focal_closer")["win"]
        .agg(["mean", "count"])
        .rename(index={0: "focal_not_closer", 1: "focal_closer"})
    )
    print()

    # Logistic regression: win ~ relative group size + relative distance.
    def fit_logit(y, X, label: str) -> None:
        print(f"Logistic regression: {label}")
        Xc = sm.add_constant(X, has_constant="add")
        try:
            model = sm.Logit(y, Xc).fit(disp=False)
            print(model.summary())
        except Exception as exc:  # noqa: BLE001
            print("Logistic regression failed:", repr(exc))
        print()

    y = df["win"]
    fit_logit(y, df[["rel_size"]], "win ~ rel_size")
    fit_logit(y, df[["rel_dist"]], "win ~ rel_dist")
    fit_logit(y, df[["rel_size", "rel_dist"]], "win ~ rel_size + rel_dist")

    # Logistic regression using binary indicators for robustness.
    fit_logit(y, df[["focal_larger"]], "win ~ focal_larger")
    fit_logit(y, df[["focal_closer"]], "win ~ focal_closer")
    fit_logit(
        y,
        df[["focal_larger", "focal_closer"]],
        "win ~ focal_larger + focal_closer",
    )


if __name__ == "__main__":
    main()
