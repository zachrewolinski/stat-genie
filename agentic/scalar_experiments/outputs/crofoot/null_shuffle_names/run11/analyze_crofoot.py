import pandas as pd
import statsmodels.api as sm


def significance_score(p):
    if p < 0.001:
        return 40
    if p < 0.01:
        return 30
    if p < 0.05:
        return 20
    if p < 0.1:
        return 10
    return 0


def main():
    df = pd.read_csv("crofoot.csv")

    group_size_focal = df["f_other"]
    group_size_other = df["win"]
    group_size_diff = group_size_focal - group_size_other

    dist_focal_home = df["m_other"]
    dist_other_home = df["n_focal"]
    loc_index = dist_other_home - dist_focal_home

    X = pd.DataFrame(
        {
            "const": 1.0,
            "group_size_diff": group_size_diff,
            "loc_index": loc_index,
        }
    )
    y = df["m_focal"]

    model = sm.Logit(y, X)
    result = model.fit(disp=False)

    p_group = result.pvalues["group_size_diff"]
    p_loc = result.pvalues["loc_index"]

    ll_full = result.llf
    ll_null = result.llnull
    pseudo_r2 = 1.0 - ll_full / ll_null if ll_null != 0 else 0.0

    support_pos = significance_score(p_group) + significance_score(p_loc)

    if support_pos > 0:
        scalar = int(round(min(100, 20 + support_pos)))
    else:
        if pseudo_r2 < 0.01:
            scalar = -80
        elif pseudo_r2 < 0.03:
            scalar = -50
        elif pseudo_r2 < 0.05:
            scalar = -20
        else:
            scalar = 0

    if scalar < -100:
        scalar = -100
    if scalar > 100:
        scalar = 100

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(int(scalar)))


if __name__ == "__main__":
    main()

