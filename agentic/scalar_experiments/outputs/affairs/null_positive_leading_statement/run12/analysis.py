import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest


def main():
    df = pd.read_csv("affairs.csv")

    # Normalize children to boolean
    children_yes = df["children"].astype(str).str.lower().str.strip() == "yes"
    affairs = pd.to_numeric(df["affairs"], errors="coerce")

    # Drop rows with missing values in key fields
    mask = affairs.notna() & children_yes.notna()
    df = df.loc[mask].copy()
    affairs = affairs.loc[mask]
    children_yes = children_yes.loc[mask]

    # Groups
    affairs_yes = affairs[children_yes]
    affairs_no = affairs[~children_yes]

    # Mean difference (no children minus children)
    mean_yes = affairs_yes.mean()
    mean_no = affairs_no.mean()
    diff_mean = mean_no - mean_yes

    # Welch's t-test
    t_stat, p_mean = stats.ttest_ind(affairs_no, affairs_yes, equal_var=False, nan_policy="omit")

    # Cohen's d (pooled std)
    n_yes = affairs_yes.shape[0]
    n_no = affairs_no.shape[0]
    s_yes = affairs_yes.std(ddof=1)
    s_no = affairs_no.std(ddof=1)
    pooled_sd = np.sqrt(((n_yes - 1) * s_yes ** 2 + (n_no - 1) * s_no ** 2) / (n_yes + n_no - 2))
    d = 0.0 if pooled_sd == 0 else diff_mean / pooled_sd

    # Any-affair proportion difference
    any_yes = (affairs_yes > 0).sum()
    any_no = (affairs_no > 0).sum()
    prop_yes = any_yes / n_yes
    prop_no = any_no / n_no
    diff_prop = prop_no - prop_yes

    count = np.array([any_no, any_yes])
    nobs = np.array([n_no, n_yes])
    z_stat, p_prop = proportions_ztest(count, nobs)

    # Heuristic scoring to Likert scale [-100, 100]
    if diff_mean > 0:
        effect_sign = 1
    elif diff_mean < 0:
        effect_sign = -1
    else:
        effect_sign = 0

    mag = min(1.0, abs(d) / 0.5)  # 0.5 ~ medium effect
    sig_mean = max(0.0, 1.0 - (p_mean / 0.05))
    sig_prop = max(0.0, 1.0 - (p_prop / 0.05))

    score = effect_sign * (40.0 * mag + 30.0 * sig_mean + 30.0 * sig_prop)
    score = int(round(max(-100, min(100, score))))

    # Write only scalar to conclusion.txt
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(score))


if __name__ == "__main__":
    main()
