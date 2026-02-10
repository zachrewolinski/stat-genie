import numpy as np
import pandas as pd


def main() -> None:
    df = pd.read_csv("caschools.csv")

    # Key variables based on metadata:
    # feature6: total enrollment (students)
    # feature7: number of teachers (FTE)
    # feature14: average reading score
    # feature15: average math score
    students = df["feature6"]
    teachers = df["feature7"].replace(0, np.nan)

    ratio = students / teachers
    score = (df["feature14"] + df["feature15"]) / 2

    mask = ratio.notna() & score.notna()
    ratio = ratio[mask].to_numpy()
    score = score[mask].to_numpy()

    if ratio.size == 0:
        scalar = 0
    else:
        # Correlation between *lower* student-teacher ratio and test scores.
        # Use -ratio so that positive correlation means: lower ratio -> higher score.
        corr = float(np.corrcoef(-ratio, score)[0, 1])

        # Map correlation strength to [-100, 100], scaled by sqrt(n).
        n = ratio.size
        base = corr  # already in [-1, 1]
        strength = base * np.sqrt(max(n, 1)) / 10.0
        strength = max(min(strength, 1.0), -1.0)
        scalar = int(round(strength * 100))

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(scalar))


if __name__ == "__main__":
    main()

