import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def main() -> None:
    # Load dataset
    df = pd.read_csv("amtl.csv")

    # Keep only the genera specified in the research question
    df = df[df["genus"].isin(["Homo sapiens", "Pan", "Papio", "Pongo"])].copy()

    # Basic cleaning: drop rows with missing key variables and ensure positive socket counts
    df = df.dropna(
        subset=["num_amtl", "sockets", "age", "prob_male", "tooth_class", "specimen"]
    )
    df = df[df["sockets"] > 0].copy()

    # Indicator for modern humans vs non-human primates
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    # Expand the data so that each individual tooth (socket) is a Bernoulli trial.
    # For each specimen–tooth-class row, we create `num_amtl` rows with AMTL=1 and
    # `sockets - num_amtl` rows with AMTL=0.
    expanded_rows: list[dict] = []
    for _, row in df.iterrows():
        n_missing = int(row["num_amtl"])
        n_present = int(row["sockets"] - row["num_amtl"])

        base = {
            "specimen": row["specimen"],
            "age": float(row["age"]),
            "prob_male": float(row["prob_male"]),
            "tooth_class": row["tooth_class"],
            "is_human": int(row["is_human"]),
            "genus": row["genus"],
            "pop": row["pop"],
        }

        for _ in range(max(n_missing, 0)):
            this = dict(base)
            this["amtl"] = 1
            expanded_rows.append(this)

        for _ in range(max(n_present, 0)):
            this = dict(base)
            this["amtl"] = 0
            expanded_rows.append(this)

    df_long = pd.DataFrame(expanded_rows)

    # Binomial logistic regression on per-tooth AMTL status.
    # Model: logit(p(AMTL)) ~ is_human + age + prob_male + tooth_class
    glm_model = smf.glm(
        formula="amtl ~ is_human + age + prob_male + C(tooth_class)",
        data=df_long,
        family=sm.families.Binomial(),
    )
    # Use cluster-robust SEs at the specimen level to account for multiple teeth per specimen
    glm_res = glm_model.fit(
        cov_type="cluster", cov_kwds={"groups": df_long["specimen"]}
    )

    coef = float(glm_res.params["is_human"])
    pval = float(glm_res.pvalues["is_human"])
    oratio = float(np.exp(coef))
    ci_low, ci_high = glm_res.conf_int().loc["is_human"].tolist()
    ci_low_or = float(np.exp(ci_low))
    ci_high_or = float(np.exp(ci_high))

    # Predicted AMTL probabilities for humans vs non-humans (averaged over the sample covariate distribution)
    df_human = df_long.copy()
    df_nonhuman = df_long.copy()
    df_human["is_human"] = 1
    df_nonhuman["is_human"] = 0
    pred_human = glm_res.predict(df_human)
    pred_nonhuman = glm_res.predict(df_nonhuman)
    mean_human = float(pred_human.mean())
    mean_nonhuman = float(pred_nonhuman.mean())
    diff = mean_human - mean_nonhuman

    # Map statistical evidence to a 0–100 scale where 0 = strong "No" and 100 = strong "Yes"
    # Here "Yes" means humans have higher AMTL frequencies than non-human primates after adjustment.
    if coef >= 0:
        # Effect in the "Yes" direction (humans higher)
        if pval < 0.001:
            score = 95
        elif pval < 0.01:
            score = 90
        elif pval < 0.05:
            score = 80
        elif pval < 0.1:
            score = 65
        else:
            # Little statistical evidence; lean slightly "No" due to lack of significance
            score = 40
    else:
        # Effect in the opposite direction (humans lower)
        if pval < 0.001:
            score = 5
        elif pval < 0.01:
            score = 10
        elif pval < 0.05:
            score = 20
        elif pval < 0.1:
            score = 35
        else:
            # Little statistical evidence; but still no support for a "Yes" answer
            score = 40

    score = int(max(0, min(100, round(score))))

    # Textual Yes/No interpretation based on coefficient direction and statistical significance
    if coef >= 0 and pval < 0.05:
        answer_label = "Yes"
    else:
        if coef < 0 and pval < 0.05:
            answer_label = "No (humans have lower AMTL frequencies than non-human primates)"
        else:
            answer_label = (
                "No (the data do not provide statistically significant evidence that "
                "humans have higher AMTL frequencies than non-human primates)"
            )

    if coef >= 0 and pval < 0.05:
        summary_clause = (
            "provide strong statistical evidence that modern humans have higher AMTL "
            "frequencies than non-human primates after adjusting for age, sex, and tooth class"
        )
    elif coef < 0 and pval < 0.05:
        summary_clause = (
            "provide strong statistical evidence that modern humans actually have lower AMTL "
            "frequencies than non-human primates after adjusting for age, sex, and tooth class"
        )
    elif pval < 0.1:
        summary_clause = (
            "are suggestive but not conventionally statistically significant, so the evidence "
            "for a difference is weak"
        )
    else:
        summary_clause = (
            "do not provide statistically significant evidence that modern humans differ in AMTL "
            "frequencies from non-human primates once age, sex, and tooth class are controlled"
        )

    explanation = (
        "I modelled the proportion of teeth missing due to antemortem tooth loss "
        "(num_amtl / sockets) using a binomial logistic regression with a logit link. "
        "The predictors were an indicator for modern humans versus non-human primates "
        "(Pan, Papio, Pongo), age-at-death, the estimated probability of being male, "
        "and tooth class (anterior, premolar, posterior). Each row in the dataset "
        "represented a specimen–tooth-class combination, and the model was weighted by "
        "the number of observable sockets, with cluster-robust standard errors at the "
        "specimen level to account for repeated measurements per individual.\n\n"
        f"The estimated coefficient for humans (is_human) was {coef:.3f}, corresponding to an "
        f"odds ratio of {oratio:.2f} (95% CI {ci_low_or:.2f}–{ci_high_or:.2f}) with a "
        f"p-value of {pval:.4g} from the cluster-robust model. Using this model, the average "
        f"predicted AMTL probability per socket was {mean_human:.3f} for humans and "
        f"{mean_nonhuman:.3f} for non-human primates, a difference of {diff:.3f} in absolute "
        "probability.\n\n"
        f"Taken together, these results {summary_clause}. Based on this analysis, my answer to "
        "the question \"Do modern humans have higher frequencies of AMTL than non-human primates, "
        "after accounting for age, sex, and tooth class?\" is "
        f"\"{answer_label}\", with strength {score}/100 on the requested Likert scale "
        "where 0 represents a strong \"No\" and 100 represents a strong \"Yes\"."
    )

    conclusion = {"response": score, "explanation": explanation}

    with open("conclusion.txt", "w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()
