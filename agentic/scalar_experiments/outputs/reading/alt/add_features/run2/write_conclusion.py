import json

conclusion = {
    "response": 10,
    "explanation": (
        "Using the dyslexic subset (dyslexia_bin=1), there are 438 observations from 73 individuals, with equal counts of reader_view on/off. "
        "Reading speed does not differ between conditions: median speed is 211.7 with reader view vs 215.1 without, and mean log-speed differs by ~0.008. "
        "Welch t-test on log-speed is non-significant (p=0.906), and Mann–Whitney on raw speed is also non-significant (p=0.732). "
        "Paired analysis across 73 participants who saw both conditions shows no effect (paired t p=0.873; Wilcoxon p=0.893). "
        "A regression controlling for page_id and num_words with cluster-robust SEs gives a near-zero reader_view coefficient (−0.0023 log units, p=0.958). "
        "Effect size is essentially zero (Cohen’s d≈0.01). Therefore, there is no evidence that Reader View improves reading speed for individuals with dyslexia in this dataset."
    ),
}

with open("conclusion.txt", "w", encoding="utf-8") as f:
    json.dump(conclusion, f)
