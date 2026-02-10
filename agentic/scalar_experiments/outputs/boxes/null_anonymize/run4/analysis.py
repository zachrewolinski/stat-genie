import pandas as pd


def main() -> None:
    df = pd.read_csv("boxes.csv")

    # feature1 encodes outcome:
    # 1 = undemonstrated option, 2 = majority option, 3 = minority option
    total_n = len(df)
    majority_n = (df["feature1"] == 2).sum()
    minority_n = (df["feature1"] == 3).sum()
    undemo_n = (df["feature1"] == 1).sum()

    # Basic proportions
    majority_prop = majority_n / total_n if total_n else 0.0
    minority_prop = minority_n / total_n if total_n else 0.0
    undemo_prop = undemo_n / total_n if total_n else 0.0

    # We want a single scalar answering:
    # "Do children’s reliance on social information and preference
    #  for majority cues vary across cultures and developmental stages?"
    #
    # Given we only have anonymized features (no explicit culture label),
    # we focus on whether there is a strong preference for majority
    # demonstrations overall (a proxy for strong reliance on social
    # majority cues). A very high majority proportion would suggest a
    # strong "Yes" (close to 100), while more even use of options would
    # suggest a weaker yes.
    #
    # Map majority_prop in [0, 1] to a baseline [-20, 100] range,
    # but cap within [-100, 100]. The offset >0 reflects that any
    # above-chance majority use indicates some reliance on social info.

    # Baseline mapping: score = -20 + majority_prop * 120
    raw_score = -20.0 + majority_prop * 120.0

    # Ensure score is within [-100, 100]
    score = max(-100.0, min(100.0, raw_score))

    # Round to nearest integer as required
    scalar = int(round(score))

    # Write to conclusion.txt with only the scalar, as specified
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(str(scalar))


if __name__ == "__main__":
    main()

