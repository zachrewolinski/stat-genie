import pandas as pd
from scipy.stats import pearsonr


def main() -> None:
    # Load data
    df = pd.read_csv("boxes.csv")

    # feature1: 1=undemonstrated, 2=majority, 3=minority
    outcome = df["feature1"]

    n = len(df)
    maj = (outcome == 2).sum()
    minr = (outcome == 3).sum()
    und = (outcome == 1).sum()

    prop_maj = maj / n
    prop_minr = minr / n
    prop_und = und / n

    # Simple across-the-board preference for majority: majority > minority & majority > undemonstrated
    _majority_pref = prop_maj > max(prop_minr, prop_und)

    # Age effects: majority choice correlation with age
    age = df["feature3"]
    r_age_maj, _p_age_maj = pearsonr(age, (outcome == 2).astype(int))

    # Very rough scalar: combine strength of majority tendency and developmental trend
    # Map majority proportion difference and correlation into [-100, 100]
    diff_maj_vs_others = prop_maj - max(prop_minr, prop_und)

    score_level = diff_maj_vs_others * 200  # if maj is 0.6 vs others 0.2 -> 0.4*200=80
    score_trend = r_age_maj * 50  # moderate positive correlation ~0.3 -> 15

    raw_score = score_level + score_trend

    # Clip to [-100, 100]
    scalar = int(round(max(-100, min(100, raw_score))))

    with open("conclusion.txt", "w") as f:
        f.write(str(scalar))


if __name__ == "__main__":
    main()

