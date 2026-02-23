import json
from pathlib import Path


def main() -> None:
    results_path = Path("analysis_results.json")
    if not results_path.exists():
        raise SystemExit("analysis_results.json not found; run analysis.py first.")

    with results_path.open() as f:
        res = json.load(f)

    avg = res["avg_preds"]
    coefs = res["coef_info_vs_homo"]

    homo = avg.get("Homo sapiens")
    pan = avg.get("Pan")
    papio = avg.get("Papio")
    pongo = avg.get("Pongo")

    pan_info = coefs.get("Pan")
    papio_info = coefs.get("Papio")
    pongo_info = coefs.get("Pongo")

    explanation = (
        "I fit a binomial logistic regression model for the proportion of missing teeth "
        "(num_amtl/sockets) using a logit link, treating sockets as binomial trials and "
        "including genus, age at death, sex estimate (prob_male), and tooth class as predictors, "
        "with Homo sapiens as the reference genus and cluster-robust standard errors by specimen. "
        f"The standardized, socket-weighted predicted probability of a tooth being missing was "
        f"approximately {homo:.3f} for Homo sapiens, {pan:.3f} for Pan, {papio:.3f} for Papio, "
        f"and {pongo:.3f} for Pongo. "
        f"Genus coefficients comparing Pan, Papio, and Pongo to Homo sapiens were "
        f"{pan_info['coef']:.2f}, {papio_info['coef']:.2f}, and {pongo_info['coef']:.2f} respectively, "
        f"with p-values {pan_info['pvalue']:.2f}, {papio_info['pvalue']:.2f}, and {pongo_info['pvalue']:.2f}; "
        "in all cases the 95% confidence intervals included zero, indicating no statistically significant "
        "differences in AMTL frequency after adjusting for age, sex, and tooth class. "
        "Moreover, the standardized predicted probabilities show chimpanzees (Pan) and baboons (Papio) with "
        "slightly higher AMTL frequencies than humans and orangutans (Pongo) with slightly lower frequencies, "
        "so humans are not consistently higher than the non-human primates. "
        "Taken together, these results provide no evidence that modern humans have higher AMTL frequencies "
        "than non-human primates once covariates are accounted for, so I answer 'No' to the research question "
        "and place my confidence near the strong-'No' end of the scale."
    )

    # Strong "No" answer given lack of significant positive differences for humans
    response_value = 10

    conclusion = {"response": response_value, "explanation": explanation}

    with Path("conclusion.txt").open("w") as f:
        json.dump(conclusion, f)


if __name__ == "__main__":
    main()
