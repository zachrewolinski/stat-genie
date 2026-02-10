import pandas as pd
from scipy import stats


def main() -> None:
    # Load data
    df = pd.read_csv("caschools.csv")

    # Construct key variables based on info.json metadata
    enrollment = df["feature6"]
    teachers = df["feature7"]
    reading = df["feature14"]
    math = df["feature15"]

    # Student-teacher ratio: students per teacher
    ratio = enrollment / teachers

    # Academic performance: average of reading and math scores
    avg_score = (reading + math) / 2.0

    # Correlations (Pearson) between ratio and scores
    corr_reading, p_reading = stats.pearsonr(ratio, reading)
    corr_math, p_math = stats.pearsonr(ratio, math)
    corr_avg, p_avg = stats.pearsonr(ratio, avg_score)

    # Simple reporting to stdout for human inspection (not used by the judge)
    print("Student-teacher ratio vs. scores")
    print(f"  Reading: corr={corr_reading:.3f}, p={p_reading:.3g}")
    print(f"  Math   : corr={corr_math:.3f}, p={p_math:.3g}")
    print(f"  Average: corr={corr_avg:.3f}, p={p_avg:.3g}")

    # Heuristic strength measure for the Likert mapping:
    # Use the average correlation magnitude across reading, math, and average.
    corrs = [corr_reading, corr_math, corr_avg]
    avg_corr = sum(corrs) / len(corrs)

    # Direction: negative correlation means lower ratio -> higher performance.
    # If correlation is positive, evidence is against the research hypothesis.
    # Map correlation (clipped to [-0.6, 0.6]) to [-100, 100] where
    # -0.6 (strongly negative) -> +100 (strong support),
    #  0   (no relationship)   -> 0   (neutral),
    # +0.6 (strongly positive) -> -100 (strong evidence against).
    max_abs_corr = 0.6
    clipped = max(-max_abs_corr, min(max_abs_corr, avg_corr))

    # Flip sign so that negative correlations support the hypothesis
    support_scaled = -clipped / max_abs_corr  # in [-1, 1]
    scalar = int(round(support_scaled * 100))

    # Write scalar to conclusion.txt as required (single integer line)
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(scalar))


if __name__ == "__main__":
    main()

