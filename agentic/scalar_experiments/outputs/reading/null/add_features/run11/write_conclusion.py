import json

conclusion = {
    "response": 30,
    "explanation": (
        "Using the dyslexia group (dyslexia_bin = 1), reading speed does not improve with Reader View. "
        "There are 438 dyslexic trials from 287 participants (reader_view: 221 off, 217 on). "
        "Mean and median speeds are lower with Reader View (mean 374 vs 407; median 278 vs 295). "
        "Welch t-test on log(speed) shows no significant difference (p=0.296, Cohen's d≈-0.10), "
        "and a Mann–Whitney test also is not significant (p=0.212). "
        "A cluster-robust regression controlling for page and word count finds a small, non-significant negative effect "
        "(reader_view coef ≈ -0.059, p=0.336). "
        "Within-subject paired analysis for participants who experienced both conditions (n=69) is also not significant "
        "(p=0.124) and the average log-speed difference is negative. "
        "Overall, there is no statistically significant evidence that Reader View improves reading speed for individuals with dyslexia."
    )
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)
