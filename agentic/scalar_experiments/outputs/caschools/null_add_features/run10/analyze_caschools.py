import pandas as pd
from scipy.stats import pearsonr


def main() -> None:
    df = pd.read_csv("caschools.csv")
    df["stratio"] = df["students"] / df["teachers"]
    df["score"] = (df["read"] + df["math"]) / 2.0
    valid = df[["stratio", "score"]].dropna()
    r, p = pearsonr(valid["stratio"], valid["score"])
    if p >= 0.05:
        scalar = 0
    else:
        scalar = int(round(max(-100.0, min(100.0, -r * 100.0))))
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(scalar))
    print(f"pearson_r={r:.4f}, p_value={p:.4g}, scalar={scalar}")


if __name__ == "__main__":
    main()

