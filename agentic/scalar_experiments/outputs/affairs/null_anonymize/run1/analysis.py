import json
import numpy as np
import pandas as pd
from math import sqrt


def cohens_h(p1: float, p2: float) -> float:
    p1 = min(max(p1, 1e-9), 1 - 1e-9)
    p2 = min(max(p2, 1e-9), 1 - 1e-9)
    return 2.0 * (np.arcsin(sqrt(p1)) - np.arcsin(sqrt(p2)))


def main():
    df = pd.read_csv("affairs.csv")
    # feature2: affair frequency, feature6: children (yes/no)
    df = df.copy()
    df["children"] = df["feature6"].astype(str).str.lower()
    df["affairs"] = pd.to_numeric(df["feature2"], errors="coerce")
    df = df.dropna(subset=["children", "affairs"])

    grp = df.groupby("children")["affairs"]
    means = grp.mean()
    counts = grp.size()
    stds = grp.std(ddof=1)

    # Proportion with any affairs (>0)
    any_affairs = df.assign(any_affairs=(df["affairs"] > 0).astype(int))
    prop = any_affairs.groupby("children")["any_affairs"].mean()

    # Expecting keys 'yes' and 'no'
    if not {"yes", "no"}.issubset(means.index):
        raise ValueError("Expected children categories 'yes' and 'no'.")

    mean_no = means.loc["no"]
    mean_yes = means.loc["yes"]
    std_no = stds.loc["no"]
    std_yes = stds.loc["yes"]
    n_no = counts.loc["no"]
    n_yes = counts.loc["yes"]

    # Pooled SD for Cohen's d
    pooled_var = (((n_no - 1) * (std_no ** 2)) + ((n_yes - 1) * (std_yes ** 2))) / (n_no + n_yes - 2)
    pooled_sd = sqrt(pooled_var) if pooled_var > 0 else np.nan
    d = (mean_no - mean_yes) / pooled_sd if pooled_sd and not np.isnan(pooled_sd) else 0.0

    p_no = prop.loc["no"]
    p_yes = prop.loc["yes"]
    h = cohens_h(p_no, p_yes)

    effect = 0.5 * d + 0.5 * h
    scalar = int(round(100 * np.tanh(effect)))
    scalar = int(max(-100, min(100, scalar)))

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(scalar))


if __name__ == "__main__":
    main()
