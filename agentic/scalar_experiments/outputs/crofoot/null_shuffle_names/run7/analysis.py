import pandas as pd
import numpy as np
import statsmodels.api as sm


def main() -> None:
    # Load data
    df = pd.read_csv("crofoot.csv")

    # Outcome: m_focal is coded 1 if focal group won, 0 otherwise
    y = df["m_focal"].astype(int)

    # Based on the metadata, f_other and win contain group sizes
    # for the focal and other groups, respectively.
    size_focal = df["f_other"]
    size_other = df["win"]
    size_diff = size_focal - size_other  # positive => focal larger

    # m_other and n_focal contain distances from home‑range centers
    # for the focal and other groups, respectively (in meters).
    dist_focal = df["m_other"]
    dist_other = df["n_focal"]
    # Positive value means the other group is farther from its center
    # than the focal group, i.e. the location favors the focal group.
    loc_adv = dist_other - dist_focal

    X = pd.DataFrame(
        {
            "size_diff": size_diff,
            "loc_adv": loc_adv,
        }
    )
    X = sm.add_constant(X)

    # Fit logistic regression
    model = sm.Logit(y, X)
    result = model.fit(disp=False)

    # Extract effects
    coef = result.params
    pvalues = result.pvalues

    size_coef = coef["size_diff"]
    loc_coef = coef["loc_adv"]
    size_p = pvalues["size_diff"]
    loc_p = pvalues["loc_adv"]

    # Simple evidence score combining standardized effect sizes
    # and significance (smaller p-values increase confidence).
    def effect_score(beta: float, p: float) -> float:
        if np.isnan(beta) or np.isnan(p):
            return 0.0
        # cap p at 0.5 to avoid extreme penalties for noisy estimates
        p_capped = min(max(p, 1e-6), 0.5)
        # log-scaled significance weight
        sig_weight = -np.log10(p_capped) / 3.0  # roughly 0–1.7 for p in [1e-6, 0.5]
        sig_weight = max(0.0, min(sig_weight, 2.0))
        # logistic transform of coefficient magnitude
        mag = 1.0 / (1.0 + np.exp(-abs(beta)))
        # direction: +1 if positive effect (supports research question),
        # -1 if negative (contradicts), 0 if ~0.
        direction = 0.0
        if beta > 0:
            direction = 1.0
        elif beta < 0:
            direction = -1.0
        return direction * mag * sig_weight

    size_score = effect_score(size_coef, size_p)
    loc_score = effect_score(loc_coef, loc_p)

    combined_score = size_score + loc_score
    # Map roughly from [-4, 4] to [-100, 100]
    scalar = int(np.clip(combined_score / 4.0 * 100.0, -100, 100))

    # Write scalar conclusion
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(int(scalar)))


if __name__ == "__main__":
    main()

