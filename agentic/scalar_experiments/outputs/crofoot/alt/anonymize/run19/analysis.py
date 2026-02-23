import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    base_dir = Path(__file__).parent

    info_path = base_dir / "info.json"
    data_path = base_dir / "crofoot.csv"
    conclusion_path = base_dir / "conclusion.txt"

    with info_path.open("r") as f:
        info = json.load(f)

    question = info.get("research_questions", [""])[0]

    df = pd.read_csv(data_path)

    df["win"] = df["feature4"].astype(int)
    df["size_diff"] = df["feature7"] - df["feature8"]
    df["home_advantage_100"] = (df["feature6"] - df["feature5"]) / 100.0

    y = df["win"]
    X = df[["size_diff", "home_advantage_100"]]
    X = sm.add_constant(X, has_constant="add")

    model = sm.Logit(y, X)
    result = model.fit(disp=False)

    params = result.params
    pvalues = result.pvalues
    odds_ratios = np.exp(params)

    n = int(df.shape[0])
    win_rate = float(df["win"].mean())

    size_p = float(pvalues["size_diff"])
    home_p = float(pvalues["home_advantage_100"])
    size_or = float(odds_ratios["size_diff"])
    home_or = float(odds_ratios["home_advantage_100"])

    def evidence_score(p: float) -> int:
        if p < 0.001:
            return 25
        if p < 0.01:
            return 20
        if p < 0.05:
            return 15
        if p < 0.1:
            return 5
        if p < 0.2:
            return -5
        return -15

    base_score = 50
    score = base_score + evidence_score(size_p) + evidence_score(home_p)
    score = max(0, min(100, score))

    direction_comments = []
    if size_or > 1:
        direction_comments.append(
            "Larger focal groups (relative to opponents) are associated with higher odds of winning."
        )
    elif size_or < 1:
        direction_comments.append(
            "Larger focal groups (relative to opponents) are associated with lower odds of winning."
        )

    if home_or > 1:
        direction_comments.append(
            "When the focal group is closer to the center of its home range than the opponent (home-range advantage), the odds of winning increase."
        )
    elif home_or < 1:
        direction_comments.append(
            "When the focal group is closer to the center of its home range than the opponent (home-range advantage), the odds of winning decrease."
        )

    significance_comments = []
    for name, p in (("relative group size", size_p), ("contest location", home_p)):
        if p < 0.05:
            significance_comments.append(
                f"There is statistically significant evidence (p = {p:.3f}) that {name} influences win probability."
            )
        elif p < 0.1:
            significance_comments.append(
                f"There is weak (marginal) evidence (p = {p:.3f}) that {name} influences win probability."
            )
        else:
            significance_comments.append(
                f"There is no strong statistical evidence (p = {p:.3f}) that {name} influences win probability in this sample."
            )

    if score >= 60:
        yes_no_statement = (
            "Overall, the analysis supports a 'Yes' answer: "
            "relative group size and/or contest location appear to influence "
            "the probability that a focal capuchin group wins an intergroup contest."
        )
    elif score <= 40:
        yes_no_statement = (
            "Overall, the analysis supports a 'No' answer: "
            "this dataset does not provide strong evidence that relative group size "
            "or contest location meaningfully influence the focal group's probability of winning."
        )
    else:
        yes_no_statement = (
            "Overall, the evidence is mixed and only weakly informative about whether "
            "relative group size and contest location influence the probability of a focal group winning."
        )

    explanation_parts = [
        f"Research question: {question}",
        f"Sample size: n = {n} contests; focal groups won {win_rate:.1%} of contests.",
        "I fit a logistic regression model with the binary outcome 'focal group won' "
        "predicted by (1) the difference in total group size (focal minus opponent) "
        "and (2) a home-range advantage term defined as how much closer the focal group "
        "is to the center of its home range than its opponent (per 100 m).",
        f"Relative group size: odds ratio = {size_or:.2f}, p-value = {size_p:.3f}.",
        f"Contest location (home-range advantage, per 100 m): odds ratio = {home_or:.2f}, p-value = {home_p:.3f}.",
        *direction_comments,
        *significance_comments,
        yes_no_statement,
        f"The scalar response on a 0–100 Likert scale (0 = strong 'No', 100 = strong 'Yes') is {score}.",
    ]

    explanation = " ".join(explanation_parts)

    conclusion_obj = {
        "response": int(score),
        "explanation": explanation,
    }

    with conclusion_path.open("w") as f:
        json.dump(conclusion_obj, f)


if __name__ == "__main__":
    main()

