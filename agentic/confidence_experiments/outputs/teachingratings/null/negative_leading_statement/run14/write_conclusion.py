import json

results = json.load(open('analysis_results.json'))

explanation = (
    f"Using {results['n']} courses, there is essentially no relationship between instructor beauty and teaching evaluations. "
    f"Pearson r = {results['pearson_r']:.4f} (p = {results['pearson_p']:.3f}) and Spearman r = {results['spearman_r']:.4f} "
    f"(p = {results['spearman_p']:.3f}) both show no association. "
    f"A simple OLS regression gives a beauty coefficient of {results['simple_coef']:.4f} (p = {results['simple_p']:.3f}), "
    f"and with controls (age, gender, minority status, course type/division, native status, tenure, and class size) the coefficient is "
    f"{results['controls_coef']:.4f} (p = {results['controls_p']:.3f}). "
    f"A 1 SD increase in beauty predicts only a {results['pred_change_1sd_beauty']:.4f} change in eval on a 1–5 scale, which is negligible. "
    "Given the consistently null, non-significant results and trivial effect sizes, the evidence supports a clear 'No'."
)

conclusion = {
    "response": 5,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)
