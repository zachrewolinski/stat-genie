import pandas as pd
import numpy as np
from scipy import stats


def main():
    df = pd.read_csv("reading.csv")

    # Reading speed is provided in feature20 (words per minute derived from words/time).
    df = df.copy()
    df["reading_speed_wpm"] = df["feature20"]

    # Dyslexia indicator (1 = dyslexia) in feature17.
    dys = df[df["feature17"] == 1].copy()
    dys = dys.dropna(subset=["feature1", "feature3", "reading_speed_wpm"])

    rv = dys[dys["feature3"] == 1]["reading_speed_wpm"]
    no = dys[dys["feature3"] == 0]["reading_speed_wpm"]

    print("Dyslexia-only observation-level comparison")
    print(f"n reader-view = {len(rv)}, n control = {len(no)}")
    print(f"mean wpm reader-view = {rv.mean():.3f}, control = {no.mean():.3f}")
    print(f"median wpm reader-view = {rv.median():.3f}, control = {no.median():.3f}")
    print("Welch t-test:", stats.ttest_ind(rv, no, equal_var=False))
    print("Mann-Whitney U:", stats.mannwhitneyu(rv, no, alternative="two-sided"))

    # Aggregate per participant to reduce repeated-measures bias.
    agg = dys.groupby(["feature1", "feature3"])["reading_speed_wpm"].mean().reset_index()
    wide = agg.pivot(index="feature1", columns="feature3", values="reading_speed_wpm")

    paired = wide.dropna()
    print("\nPer-participant paired comparison (participants with both conditions)")
    print(f"paired n = {len(paired)}")
    if len(paired) >= 2:
        diff = paired[1] - paired[0]
        print(f"mean diff (reader-view - control) = {diff.mean():.3f}")
        print(f"median diff = {diff.median():.3f}")
        print("Paired t-test:", stats.ttest_rel(paired[1], paired[0]))
        print("Wilcoxon signed-rank:", stats.wilcoxon(diff))

    # Also show per-participant (unpaired) comparison of means by condition.
    rv_part = wide[1].dropna()
    no_part = wide[0].dropna()
    print("\nPer-participant mean comparison (unpaired)")
    print(f"n reader-view participants = {len(rv_part)}, n control participants = {len(no_part)}")
    print(f"mean wpm reader-view = {rv_part.mean():.3f}, control = {no_part.mean():.3f}")
    print("Welch t-test:", stats.ttest_ind(rv_part, no_part, equal_var=False))


if __name__ == "__main__":
    main()
