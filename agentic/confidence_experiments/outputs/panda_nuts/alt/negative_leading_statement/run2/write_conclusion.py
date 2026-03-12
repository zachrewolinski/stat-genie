import json

with open('analysis_results.json', 'r') as f:
    res = json.load(f)

p = res['primary']
r = res['hammer_control']

explanation = (
    "Using 84 sessions from 22 chimpanzees, I defined efficiency as nuts_opened/seconds "
    f"(mean {p['efficiency_mean']:.3f} nuts/sec). I fit an OLS model with cluster-robust SEs "
    "by chimpanzee. Age showed a positive association with efficiency "
    f"(coef {p['age_coef']:.3f} nuts/sec per year, 95% CI {p['age_ci_low']:.3f} to {p['age_ci_high']:.3f}, "
    f"p={p['age_p']:.4f}). Males were more efficient than females "
    f"(coef {p['sex_m_coef']:.3f} nuts/sec, p={p['sex_m_p']:.4f}). Receiving help was negative but not "
    f"statistically significant in the primary model (coef {p['help_y_coef']:.3f}, p={p['help_y_p']:.4f}). "
    "As a robustness check including hammer type, help became significantly negative "
    f"(p={r['help_y_p']:.4f}), while age was borderline (p={r['age_p']:.4f}) and sex remained significant "
    f"(p={r['sex_m_p']:.4f}). Overall, the evidence supports that age and sex influence nut-cracking efficiency, "
    "with weaker/conditional evidence for help, so a moderate 'Yes' is warranted."
)

out = {
    "response": 68,
    "explanation": explanation,
}

with open('conclusion.txt', 'w') as f:
    json.dump(out, f)
