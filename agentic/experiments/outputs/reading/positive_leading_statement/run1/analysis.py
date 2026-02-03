import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import ttest_ind


def main() -> None:
    df = pd.read_csv("reading.csv")

    # Focus on readers with dyslexia
    dys = df[df["dyslexia_bin"] == 1].copy()
    dys = dys.dropna(subset=["speed", "reader_view"])

    # Group summary
    summary = (
        dys.groupby("reader_view")["speed"]
        .agg(["count", "mean", "median", "std"])
        .rename(index={0: "no_reader_view", 1: "reader_view"})
    )

    # Welch t-test on raw speed
    rv1 = dys[dys["reader_view"] == 1]["speed"]
    rv0 = dys[dys["reader_view"] == 0]["speed"]
    t_stat, p_val, _ = ttest_ind(rv1, rv0, usevar="unequal")

    # Log-speed regression with basic controls
    reg_data = dys.dropna(
        subset=["speed", "reader_view", "page_id", "device", "num_words", "Flesch_Kincaid"]
    ).copy()
    reg_data["log_speed"] = np.log(reg_data["speed"])
    model = smf.ols(
        "log_speed ~ reader_view + C(page_id) + C(device) + num_words + Flesch_Kincaid",
        data=reg_data,
    ).fit(cov_type="HC3")

    print("Dyslexia sample size:", len(dys))
    print("\nSpeed summary (dyslexia only):")
    print(summary)
    print("\nWelch t-test on speed (reader_view - no_reader_view):")
    print(f"t_stat={t_stat:.4f}, p_value={p_val:.4f}")
    print("\nLog-speed regression (HC3 robust):")
    print(
        f"reader_view coef={model.params['reader_view']:.6f}, "
        f"p_value={model.pvalues['reader_view']:.4f}"
    )


if __name__ == "__main__":
    main()
