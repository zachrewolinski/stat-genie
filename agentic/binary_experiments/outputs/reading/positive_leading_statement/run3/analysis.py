import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.weightstats import ttest_ind


def main():
    df = pd.read_csv("reading.csv")

    # Focus on participants with dyslexia
    dys = df[df["dyslexia_bin"] == 1].copy()

    # Drop rows with missing speed or reader_view
    dys = dys.dropna(subset=["speed", "reader_view", "num_words", "adjusted_running_time"])

    # Summary stats
    summary = dys.groupby("reader_view")["speed"].agg(["count", "mean", "median", "std"]).reset_index()

    # T-test on raw speed
    speed_rv = dys[dys["reader_view"] == 1]["speed"]
    speed_no = dys[dys["reader_view"] == 0]["speed"]
    t_stat, p_val, dfree = ttest_ind(speed_rv, speed_no, usevar="unequal")

    # Log-speed robustness (to reduce skew)
    dys = dys[dys["speed"] > 0].copy()
    dys["log_speed"] = np.log(dys["speed"])
    log_rv = dys[dys["reader_view"] == 1]["log_speed"]
    log_no = dys[dys["reader_view"] == 0]["log_speed"]
    t_stat_log, p_val_log, dfree_log = ttest_ind(log_rv, log_no, usevar="unequal")

    # Regression controlling for text difficulty and device etc.
    # Use log_speed as dependent variable
    # Categorical controls: device, language, education, english_native, gender
    # Numeric controls: age, num_words, Flesch_Kincaid, img_width, correct_rate, retake_trial
    model_df = dys.dropna(subset=[
        "log_speed",
        "reader_view",
        "device",
        "language",
        "education",
        "english_native",
        "gender",
        "age",
        "num_words",
        "Flesch_Kincaid",
        "img_width",
        "correct_rate",
        "retake_trial",
    ]).copy()

    # Build design matrix with patsy via statsmodels formula
    import statsmodels.formula.api as smf

    formula = (
        "log_speed ~ reader_view + age + num_words + Flesch_Kincaid + img_width + correct_rate + retake_trial "
        "+ C(device) + C(language) + C(education) + C(english_native) + C(gender)"
    )
    reg = smf.ols(formula, data=model_df).fit(cov_type="HC3")

    # Save key outputs
    with open("analysis_results.txt", "w") as f:
        f.write("Dyslexia-only sample size: %d\n" % len(dys))
        f.write("\nSpeed summary by reader_view (0=off,1=on):\n")
        f.write(summary.to_string(index=False))
        f.write("\n\nT-test raw speed (unequal var): t=%.4f, p=%.6f, df=%.1f\n" % (t_stat, p_val, dfree))
        f.write("T-test log(speed) (unequal var): t=%.4f, p=%.6f, df=%.1f\n" % (t_stat_log, p_val_log, dfree_log))
        f.write("\nRegression (log_speed) with controls, robust SE (HC3):\n")
        f.write(reg.summary().as_text())


if __name__ == "__main__":
    main()
