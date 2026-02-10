import pathlib

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_data(csv_path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def build_variables(df: pd.DataFrame) -> pd.DataFrame:
    # Outcome: 1 if focal group won, 0 otherwise
    y = df["m_focal"].astype(int)

    # According to info.json descriptions (names are shuffled):
    # f_other: number of individuals in focal group
    # win:    number of individuals in other group
    size_focal = df["f_other"]
    size_other = df["win"]
    size_adv = size_focal - size_other

    # m_other: distance of focal group from center of its home range
    # n_focal: distance of other group from center of its home range
    dist_focal_center = df["m_other"]
    dist_other_center = df["n_focal"]
    # Positive when focal group is closer to its own center than the other group is to its center
    loc_adv = dist_other_center - dist_focal_center

    X = pd.DataFrame(
        {
            "size_adv": size_adv,
            "loc_adv": loc_adv,
        }
    )

    return pd.concat([y.rename("win_focal"), X], axis=1)


def fit_logistic_model(df_vars: pd.DataFrame):
    y = df_vars["win_focal"]
    X = df_vars[["size_adv", "loc_adv"]]
    X = sm.add_constant(X, has_constant="add")
    model = sm.Logit(y, X)
    result = model.fit(disp=False)
    return result


def evidence_score(p: float) -> int:
    """Map a p-value to an evidence strength score in [0, 50]."""
    if p < 1e-4:
        return 50
    if p < 1e-3:
        return 45
    if p < 1e-2:
        return 35
    if p < 5e-2:
        return 25
    if p < 1e-1:
        return 10
    if p < 2e-1:
        return 5
    return 0


def compute_scalar_conclusion(result) -> int:
    pvals = result.pvalues

    p_size = float(pvals.get("size_adv", np.nan))
    p_loc = float(pvals.get("loc_adv", np.nan))

    score_size = evidence_score(p_size) if np.isfinite(p_size) else 0
    score_loc = evidence_score(p_loc) if np.isfinite(p_loc) else 0

    base_score = score_size + score_loc

    # If there is essentially no evidence for either predictor (large p-values),
    # interpret this as modest evidence against an effect.
    if base_score == 0 and p_size > 0.5 and p_loc > 0.5:
        scalar = -30
    else:
        scalar = base_score

    # Clamp to Likert range [-100, 100]
    scalar = int(max(-100, min(100, round(scalar))))
    return scalar


def main():
    cwd = pathlib.Path(__file__).resolve().parent
    csv_path = cwd / "crofoot.csv"

    df = load_data(csv_path)
    df_vars = build_variables(df)
    result = fit_logistic_model(df_vars)

    scalar = compute_scalar_conclusion(result)

    conclusion_path = cwd / "conclusion.txt"
    conclusion_path.write_text(f"{scalar}\n", encoding="utf-8")


if __name__ == "__main__":
    main()

